// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================

#include <algorithm>
#include <cctype>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <boost/program_options.hpp>

#include <ivqML/IO/CSV.h>
#include <ivqML/Model/Regression/Logistic.h>
#include <ivqML/Optimizer/Adam.h>
#include <ivqML/Optimizer/GradientDescent.h>
#include <ivqML/Optimizer/Debug/Simple.h>

using TNatural = unsigned long long;
using TReal = long double;

/**
 */
struct Args
{
  Args( int argc, char** argv );
  void show_error( std::ostream& o );
  bool fail( );

  int Success;
  std::stringstream Error;

  TReal Alpha { 1e-2 };
  TReal L1 { 0 };
  TReal L2 { 0 };
  TNatural BatchSize { 0 };
  TNatural Epochs { 1000 };

  std::string Optimizer     { "Adam" };
  std::string Validation    { "MCE" };
  std::string Test          { "0.3" };
  std::string TrainFilename { "" };
  char Delimiter            { ',' };
};

// -------------------------------------------------------------------------
std::string ToLower( const std::string& s );

template< class _TOptimizer >
void fit( const Args& args );

// -------------------------------------------------------------------------
int main( int argc, char** argv )
{
  using TModel = ivqML::Model::Regression::Logistic< TReal, TNatural >;

  Args args( argc, argv );
  if( args.fail( ) )
  {
    args.show_error( std::cerr );
    return( EXIT_FAILURE );
  } // end if

  if( ToLower( args.Optimizer ) == "gradientdescent" )
    fit< ivqML::Optimizer::GradientDescent< TModel > >( args );
  else if( ToLower( args.Optimizer ) == "adam" )
    fit< ivqML::Optimizer::Adam< TModel > >( args );
  else
  {
    std::cerr
      << "Invalid optimizer \"" << args.Optimizer << "\""
      << std::endl;
    return( EXIT_FAILURE );
  } // end if

  return( EXIT_SUCCESS );
}

// -------------------------------------------------------------------------
Args::
Args( int argc, char** argv )
{
  namespace PO = boost::program_options;
  try
  {
    PO::options_description desc( "Logistic regression" );
    desc.add_options( )
      ( "help,h", "this message" )
      ( "alpha,a", PO::value< TReal >( &this->Alpha )->default_value( this->Alpha ), "Learning rate" )
      ( "L1", PO::value< TReal >( &this->L1 )->default_value( this->L1 ), "LASSO coefficient" )
      ( "L2", PO::value< TReal >( &this->L2 )->default_value( this->L2 ), "Ridge coefficient" )
      ( "epochs,e", PO::value< TNatural >( &this->Epochs )->default_value( this->Epochs ), "Maximum numer of epochs" )
      ( "batch_size,b", PO::value< TNatural >( &this->BatchSize )->default_value( this->BatchSize ), "Batch size" )
      ( "optimizer,o", PO::value< std::string >( &this->Optimizer )->default_value( this->Optimizer ), "Adam|GradientDescent" )
      ( "validation,v", PO::value< std::string >( &this->Validation )->default_value( this->Validation ), "MCE|LOO|KFoldK" )
      ( "test,t", PO::value< std::string >( &this->Test )->default_value( this->Test ), "test_size|test_filename" )
      ( "delimiter,d", PO::value< char >( &this->Delimiter )->default_value( this->Delimiter ), "CSV field delimiter" )
      ;

    PO::positional_options_description pos;
    desc.add_options( )
      ( "train_filename", PO::value< std::string >( &this->TrainFilename ), "CSV file with training data" )
      ;
    pos.add( "train_filename", 1 );

    PO::variables_map vm;
    // PO::store( PO::parse_command_line( argc, argv, desc ), vm );
    PO::store( PO::command_line_parser( argc, argv ).options( desc ).positional( pos ).run( ), vm );
    PO::notify( vm );

    this->Error << "";
    this->Success = 0;

    if( argc == 1 || vm.count( "help" ) > 0 )
    {
      this->Error << desc;
      this->Success = 1;
      return;
    } // end if

    if( vm.count( "train_filename" ) == 0 )
    {
      this->Error << "Give, at least, a CSV file with training data.";
      this->Success = 2;
      return;
    } // end if
  }
  catch( const std::exception& err )
  {
    this->Error << err.what( );
    this->Success = 3;
    return;
  } // end ry
}

// -------------------------------------------------------------------------
void Args::
show_error( std::ostream& o )
{
  o << this->Error.str( ) << std::endl;
}

// -------------------------------------------------------------------------
bool Args::
fail( )
{
  return( this->Success != 0 );
}

// -------------------------------------------------------------------------
std::string ToLower( const std::string& s )
{
  std::string r = s;
  std::transform(
    r.begin( ), r.end( ), r.begin( ),
    []( const unsigned char& c ){ return( std::tolower( c ) ); }
    );
  return( r );
}

// -------------------------------------------------------------------------
template< class _TOptimizer >
void fit( const Args& args )
{
  using TOptimizer = _TOptimizer;
  using TModel = typename TOptimizer::TModel;
  using TNatural = typename TOptimizer::TNatural;
  using TReal = typename TOptimizer::TReal;
  using TMatrix = typename TOptimizer::TMatrix;

  TMatrix X_tr( 0, 0 ), y_tr( 0, 0 ), X_te( 0, 0 ), y_te( 0, 0 );

  // Prepare test data
  TReal test_coeff;
  std::istringstream test_istr( args.Test );
  test_istr >> test_coeff;
  if( test_istr.eof( ) && !test_istr.fail( ) )
  {
    // Read training data
    TMatrix D;
    ivqML::IO::ReadCSV( D, args.TrainFilename, 0, args.Delimiter );

    // Split testing data
    TReal train_coeff = std::fabs( TReal( 1 ) - std::fabs( test_coeff ) );
    if( train_coeff > 1 ) train_coeff = TReal( 1 );

    struct
    {
      void init(
        const TReal& v, const Eigen::Index& r, const Eigen::Index& c
        )
        {
          this->Z.clear( );
          this->O.clear( );
          this->operator()( v, r, c );
        }
      void operator()(
        const TReal& v, const Eigen::Index& r, const Eigen::Index& c
        )
        {
          if     ( v == 0 ) this->Z.push_back( r );
          else if( v == 1 ) this->O.push_back( r );
        }
      std::vector< Eigen::Index > Z, O;
    } zo_visit;
    D.col( D.cols( ) - 1 ).visit( zo_visit );

    // Shuffle
    std::random_device rand_dev;
    std::mt19937 rang_gen( rand_dev( ) );
    std::shuffle( zo_visit.Z.begin( ), zo_visit.Z.end( ), rang_gen );
    std::shuffle( zo_visit.O.begin( ), zo_visit.O.end( ), rang_gen );

    TNatural n = std::min( zo_visit.Z.size( ), zo_visit.O.size( ) );
    TNatural n_tr = TNatural( TReal( n ) * train_coeff );
    X_tr = D.block( 0, 0, n_tr, D.cols( ) - 1 );
    y_tr = D.block( 0, D.cols( ) - 1, n_tr, 1 );
    if( n_tr < n )
    {
      X_te = D.block( n_tr, 0, n - n_tr, D.cols( ) - 1 );
      y_te = D.block( n_tr, D.cols( ) - 1, n - n_tr, 1 );
    } // end if
  }
  else
  {
    std::cout << "uh oh" << std::endl;
  } // end if

  // Prepare model
  TModel m( X_tr.cols( ) );
  std::cout << "Init model: " << m << std::endl;

  // Prepare debugger
  ivqML::Optimizer::Debug::Simple< TModel > debug( m, std::cout, args.Epochs );

  // Prepare optimizer
  TOptimizer opt( m );
  opt.setAlpha( args.Alpha );
  opt.setLambda1( args.L1 );
  opt.setLambda2( args.L2 );
  opt.setDebug( debug );

  // Fit model to training and testing data
  opt.fit( X_tr, y_tr, X_te, y_te );

  // Show results
  std::cout << "Fitted model: " << m << std::endl;
  std::cout << "Training cost: " << m.cost( X_tr, y_tr ) << std::endl;
  if( X_te.rows( ) > 0 )
    std::cout << "Testing cost: " << m.cost( X_te, y_te ) << std::endl;

  // Confussion matrices
  TMatrix z = m( X_tr, true );
  TMatrix y_obs( y_tr.rows( ), 2 ), y_pre( y_tr.rows( ), 2 );
  y_obs << TReal( 1 ) - y_tr.array( ), y_tr;
  y_pre << TReal( 1 ) - z.array( ), z;

  std::cout << ( y_obs.transpose( ) * y_pre ) << std::endl;

}

// eof - $RCSfile$
