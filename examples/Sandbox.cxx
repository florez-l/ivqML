// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================

#include <iostream>
#include <string>
#include <ivq/eigen/Config.h>
#include <ivqML/IO/CSV.h>


#include <algorithm>
#include <functional>
#include <numeric>
#include <random>

#include <vtkChartXY.h>
#include <vtkContextView.h>
#include <vtkFloatArray.h>
#include <vtkNamedColors.h>
#include <vtkNew.h>
#include <vtkPlotPoints.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkRenderer.h>
#include <vtkTable.h>




namespace ivqML
{
  namespace Common
  {
    namespace KMeans
    {
      /**
       */
      template< class _TM, class _TX >
      void RandomInit(
        Eigen::EigenBase< _TM >& _m, const Eigen::EigenBase< _TX >& _X
        )
      {
        using _R = typename _TM::Scalar;

        std::vector< unsigned long long > I( _X.rows( ) );
        std::iota( I.begin( ), I.end( ), 0 );
        std::shuffle(
          I.begin( ), I.end( ),
          std::default_random_engine(
            std::chrono::system_clock::now( ).time_since_epoch( ).count( )
            )
          );
        I.erase( I.begin( ) + _m.rows( ), I.end( ) );
        _m.derived( ) = _X.derived( )( I, ivq_EIGEN_ALL ).template cast< _R >( );
      }

      /**
       */
      template< class _TM, class _TX >
      void Fit(
        Eigen::EigenBase< _TM >& _m, const Eigen::EigenBase< _TX >& _X,
        std::function< bool( const typename _TM::Scalar&, const unsigned long long& ) > debug
        =
        []( const typename _TM::Scalar&, const unsigned long long& ) -> bool { return( false ); }
        )
      {
        using _R = typename _TM::Scalar;
        using _M = Eigen::Matrix< _R, Eigen::Dynamic, Eigen::Dynamic >;
        using _L = Eigen::Matrix< unsigned short, Eigen::Dynamic, 1 >;

        static const _R eps = std::numeric_limits< _R >::epsilon( );

        auto X = _X.derived( ).template cast< _R >( );
        auto& m = _m.derived( );
        unsigned long long F = m.cols( );
        unsigned long long K = m.rows( );
        unsigned long long N = X.rows( );

        _M D( N, K ), J( N, 1 ), P = m;
        _L L( N );

        unsigned long long iter = 0;
        bool stop = false;
        while( !stop )
        {
          // Update distances
          for( unsigned long long k = 0; k < K; ++k )
            D.col( k ) = ( X.rowwise( ) - m.row( k ) ).array( ).pow( 2 ).rowwise( ).sum( ).sqrt( );

          // Update labels
          for( unsigned long long n = 0; n < N; ++n )
            D.row( n ).minCoeff( &L( n ) );

          // Update means
          for( unsigned long long k = 0; k < K; ++k )
          {
            J = ( L.array( ) == k ).template cast< _R >( );
            unsigned long long j = J.sum( );
            if( j > 0 )
              m.row( k ) = ( X.array( ).colwise( ) * J.col( 0 ).array( ) ).colwise( ).sum( ) / _R( j );
            else
              m.row( k ).array( ) *= _R( 0 );
          } // end for

          // Stop criterion
          iter++;
          _R mse = ( P - m ).array( ).pow( 2 ).rowwise( ).mean( ).mean( );
          stop = debug( mse, iter ) || !( eps < mse );
          P = m;
        } // end while
      }
    } // end namespace
  } // end namespace
} // end namespace

using TReal = float;
using TMatrix = Eigen::Matrix< TReal, Eigen::Dynamic, Eigen::Dynamic >;

int main( int argc, char** argv )
{
  std::string input_fn = argv[ 1 ];
  unsigned int K = 3;

  TMatrix D;
  if( !ivqML::IO::CSV::Read( D, input_fn, 1, ',' ) )
  {
    std::cerr << "Error reading \"" << input_fn << "\"" << std::endl;
    return( EXIT_FAILURE );
  } // end if
  auto X = D.block( 0, 0, D.rows( ), D.cols( ) - 1 );
  TMatrix means( K, X.cols( ) );
  ivqML::Common::KMeans::RandomInit( means, X );

  // Create a table with some points in it
  vtkNew< vtkFloatArray > arrX, arrY, meansX, meansY;
  arrX->SetName( "X Axis" );
  arrY->SetName( "Y Axis" );
  meansX->SetName( "X means" );
  meansY->SetName( "Y means" );

  arrX->SetVoidArray( D.data( ), D.rows( ), 1 );
  arrY->SetVoidArray( D.data( ) + D.rows( ), D.rows( ), 1 );

  meansX->SetVoidArray( means.data( ), means.rows( ), 1 );
  meansY->SetVoidArray( means.data( ) + means.rows( ), means.rows( ), 1 );

  vtkNew< vtkTable > table;
  table->AddColumn( arrX );
  table->AddColumn( arrY );
  table->SetNumberOfRows( D.rows( ) );

  vtkNew< vtkTable > tableMeans;
  tableMeans->AddColumn( meansX );
  tableMeans->AddColumn( meansY );
  tableMeans->SetNumberOfRows( means.rows( ) );

  vtkNew< vtkNamedColors > colors;

  // Set up a 2D scene, add an XY chart to it.
  vtkNew< vtkContextView > view;
  view->GetRenderWindow( )->SetSize( 1024, 768 );
  view->GetRenderWindow( )->SetWindowName( "ScatterPlot" );
  view->GetRenderer( )->SetBackground( colors->GetColor3d( "SlateGray" ).GetData( ) );

  vtkNew< vtkChartXY > chart;
  view->GetScene( )->AddItem( chart );
  chart->SetShowLegend( false );

  // Add multiple scatter plots, setting the colors etc.
  vtkPlot* points = chart->AddPlot( vtkChart::POINTS );
  points->SetInputData( table, 0, 1 );
  points->SetColor( 0, 0, 0, 255 );
  points->SetWidth( 0.1 );
  dynamic_cast< vtkPlotPoints* >( points )->SetMarkerStyle( vtkPlotPoints::PLUS );

  points = chart->AddPlot( vtkChart::POINTS );
  points->SetInputData( tableMeans, 0, 1 );
  points->SetColor( 255, 0, 0, 255 );
  points->SetWidth( 1.5 );
  dynamic_cast< vtkPlotPoints* >( points )->SetMarkerStyle( vtkPlotPoints::CIRCLE );

  // Finally render the scene.
  view->GetRenderWindow( )->SetMultiSamples( 0 );
  view->GetRenderWindow( )->Render( );
  view->GetInteractor( )->Initialize( );


  /* TODO
     ivqML::Common::KMeans::Fit(
     means, X,
     [&means]( const TReal& err, const unsigned long long& iter ) -> bool
     {
     std::cout << iter << " " << err << std::endl;
     return( false );
     }
     );
  */




  view->GetInteractor( )->Start( );

  return( EXIT_SUCCESS );


  /* TODO
     std::string input_fn = argv[ 1 ];
     unsigned int K = 3;

     TMatrix D;
     if( ivqML::IO::CSV::Read( D, input_fn, 1, ',' ) )
     {
     auto X = D.block( 0, 0, D.rows( ), D.cols( ) - 1 );

     TMatrix means( K, X.cols( ) );
     ivqML::Common::KMeans::RandomInit( means, X );
     std::cout << "***************************" << std::endl;
     std::cout << means << std::endl;
     std::cout << "***************************" << std::endl;

     ivqML::Common::KMeans::Fit(
     means, X,
     [&means]( const TReal& err, const unsigned long long& iter ) -> bool
     {
     std::cout << iter << " " << err << std::endl;
     return( false );
     }
     );

     std::cout << "***************************" << std::endl;
     std::cout << means << std::endl;
     std::cout << "***************************" << std::endl;
     }
     else
     {
     std::cerr << "Error reading \"" << input_fn << "\"" << std::endl;
     return( EXIT_FAILURE );
     } // end if
  */
}


/* TODO
   #include <iostream>
   #include <sstream>

   #include <itkImageFileWriter.h>
   #include <itkRescaleIntensityImageFilter.h>
   #include <itkVectorImage.h>
   #include <ivq/ITK/ImageFileReader.h>
   #include <ivqML/ITK/MixtureOfGaussiansImageFilter.h>

   const unsigned int Dim = 2;
   using TReal = float;
   using TImage = itk::VectorImage< TReal, Dim >;

   int main( int argc, char** argv )
   {
   if( argc < 3 )
   {
   std::cerr
   << "Usage: " << argv[ 0 ]  << " input_image output_image [K=2]"
   << std::endl;
   return( EXIT_FAILURE );
   } // end if
   std::string input_image = argv[ 1 ];
   std::string output_image = argv[ 2 ];
   unsigned long long K = 2;
   if( argc > 3 ) std::istringstream( argv[ 3 ] ) >> K;

   auto reader = ivq::ITK::ImageFileReader< TImage >::New( );
   reader->SetFileName( input_image );

   auto mog = ivqML::ITK::MixtureOfGaussiansImageFilter< TImage >::New( );
   mog->SetInput( reader->GetOutput( ) );
   mog->SetNumberOfMeans( K );
   mog->SetDebug(
   []( const TReal& mse ) -> bool
   {
   std::cout << "MSE = " << mse << std::endl;
   return( false );
   }
   );

   using TLabels = decltype( mog )::ObjectType::TOutImage;
   using TLabel = itk::NumericTraits< TLabels::PixelType >::ValueType;

   auto rescaler = itk::RescaleIntensityImageFilter< TLabels, TLabels >::New( );
   rescaler->SetInput( mog->GetOutput( ) );
   rescaler->SetOutputMinimum( std::numeric_limits< TLabel >::min( ) );
   rescaler->SetOutputMaximum( std::numeric_limits< TLabel >::max( ) );

   auto writer = itk::ImageFileWriter< TLabels >::New( );
   writer->SetInput( rescaler->GetOutput( ) );
   writer->SetFileName( output_image );
   try
   {
   writer->Update( );
   }
   catch( const std::exception& err )
   {
   std::cerr << "Error caught: " << err.what( ) << std::endl;
   return( EXIT_FAILURE );
   } // end try

   return( EXIT_SUCCESS );
   }
*/

// eof - $RCSfile$
