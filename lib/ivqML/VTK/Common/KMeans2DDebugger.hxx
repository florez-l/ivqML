// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __ivqML__VTK___Common__KMeans2DDebugger__hxx__
#define __ivqML__VTK___Common__KMeans2DDebugger__hxx__

#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkRenderer.h>

// -------------------------------------------------------------------------
template< class _TX, class _TM >
void ivqML::VTK::Common::KMeans2DDebugger< _TX, _TM >::
Init( Eigen::EigenBase< _TX >& X, ::Eigen::EigenBase< _TM >& M )
{
  this->_X = &( X.derived( ) );
  this->_M = &( M.derived( ) );

  this->_Xtable.Init( *( this->_X ) );
  this->_Mtable.Init( *( this->_M ) );

  this->_Colors = vtkSmartPointer< vtkNamedColors >::New( );
  this->_View = vtkSmartPointer< vtkContextView >::New( );
  this->_Chart = vtkSmartPointer< vtkChartXY >::New( );

  this->_View->GetRenderWindow( )->SetSize( 1024, 768 );
  this->_View->GetRenderWindow( )->SetWindowName( "2D KMeans debugger" );
  this->_View->GetRenderer( )
    ->SetBackground( this->_Colors->GetColor3d( "SlateGray" ).GetData( ) );
  this->_View->GetScene( )->AddItem( this->_Chart );
  this->_Chart->SetShowLegend( false );

  this->_Xpoints = this->_Chart->AddPlot( vtkChart::POINTS );
  this->_Xpoints->SetInputData( this->_Xtable._Table, 0, 1 );
  this->_Xpoints->SetColor( 0, 0, 0, 255 );
  this->_Xpoints->SetWidth( 0.1 );
  dynamic_cast< vtkPlotPoints* >( this->_Xpoints )
    ->SetMarkerStyle( vtkPlotPoints::PLUS );
  
  this->_Mpoints = this->_Chart->AddPlot( vtkChart::POINTS );
  this->_Mpoints->SetInputData( this->_Mtable._Table, 0, 1 );
  this->_Mpoints->SetColor( 255, 0, 0, 255 );
  this->_Mpoints->SetWidth( 1.5 );
  dynamic_cast< vtkPlotPoints* >( this->_Mpoints )
    ->SetMarkerStyle( vtkPlotPoints::CIRCLE );

  // Prepare visualization
  this->_View->GetRenderWindow( )->Render( );
  this->_View->GetInteractor( )->Initialize( );
}

// -------------------------------------------------------------------------
template< class _TX, class _TM >
bool ivqML::VTK::Common::KMeans2DDebugger< _TX, _TM >::
operator()( const float& err, const unsigned long long& iter )
{
  std::cout << iter << " " << err << std::endl;
  this->_Mtable.Modified( );
  this->_Mpoints->Modified( );
  this->_Chart->Modified( );
  this->_View->Modified( );

  this->_Mpoints->Update( );
  this->_Chart->Update( );
  this->_View->Update( );
  this->_View->Render( );

  std::this_thread::sleep_for( std::chrono::milliseconds( 500 ) );

  return( false );
}

// -------------------------------------------------------------------------
template< class _TX, class _TM >
void ivqML::VTK::Common::KMeans2DDebugger< _TX, _TM >::
Start( )
{
  this->_View->GetInteractor( )->Start( );
}

#endif // __ivqML__VTK___Common__KMeans2DDebugger__hxx__

// eof - $RCSfile$
