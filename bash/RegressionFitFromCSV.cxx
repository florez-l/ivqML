// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================

#include <algorithm>
#include <iostream>
#include <string>
#include <boost/program_options.hpp>
#include <ivqML/Model/Regression/Linear.h>
#include <ivqML/Cost/MeanSquareError.h>
#include <ivqML/Model/Regression/Logistic.h>
#include <ivqML/Cost/BinaryCrossEntropy.h>
#include <ivqML/Optimizer/ADAM.h>
#include <ivqML/Optimizer/GradientDescent.h>

// -------------------------------------------------------------------------
using TScalar = long double;

// -------------------------------------------------------------------------
#define PROGRAM_OPTIONS_OPTION( _O, _D )                \
  ( _D, boost::program_options::value< decltype( _O ) > \
    ( &_O )->default_value( _O ), "" )

// -------------------------------------------------------------------------
template< class _TOptimizer >
bool fit( int argc, char** argv )
{
  std::cout << "_Z" << typeid( _TOptimizer ).name( ) << std::endl;

  return( false );
}

// -------------------------------------------------------------------------
template< class _TCost >
bool choose_optimizer( int argc, char** argv )
{
  unsigned int optimizer
    =
    std::distance(
      argv + 1, std::find( argv + 1, argv + argc, "--optimizer" )
      );
  if( optimizer < argc )
  {
    if( std::string( argv[ optimizer ] ) == "gd" )
      return(
        fit< ivqML::Optimizer::GradientDescent< _TCost > >( argc, argv )
        );
    else if( std::string( argv[ optimizer ] ) == "ADAM" )
      return(
        fit< ivqML::Optimizer::ADAM< _TCost > >( argc, argv )
        );
    else
    {
      std::cerr
        << "Usage: " << argv[ 0 ]
        << " ... --optimizer [gd/ADAM] ..."
        << std::endl;
      return( false );
    } // end if
  }
  else
    return( fit< ivqML::Optimizer::ADAM< _TCost > >( argc, argv ) );

  /* TODO
     if( optimizer == "gd" )
     {
     return( fit< ivqML::Optimizer::GradientDescent< _TCost > >( opt, csv ) );
     }
     else if( optimizer == "ADAM" )
     {
     return( fit< ivqML::Optimizer::ADAM< _TCost > >( opt, csv ) );
     }
     else
     {
     std::cerr << opt << std::endl;
     std::cerr << "******** POSSIBLE OPTIMIZERS ********" << std::endl;
     std::cerr << "gd,ADAM" << std::endl;
     std::cerr << "*************************************" << std::endl;
     return( false );
     } // end if
  */
  return( false );
}

// -------------------------------------------------------------------------
bool linear_fit( int argc, char** argv )
{
  using TModel = ivqML::Model::Regression::Linear< TScalar >;
  using TCost = ivqML::Cost::MeanSquareError< TModel >;
  return( choose_optimizer< TCost >( argc, argv ) );
}

// -------------------------------------------------------------------------
bool logistic_fit( int argc, char** argv )
{
  using TModel = ivqML::Model::Regression::Logistic< TScalar >;
  using TCost = ivqML::Cost::BinaryCrossEntropy< TModel >;
  return( choose_optimizer< TCost >( argc, argv ) );
}

// -------------------------------------------------------------------------
int main( int argc, char** argv )
{
  unsigned int model
    =
    std::distance(
      argv + 1,
      std::find_if(
        argv + 1, argv + argc,
        []( char* s ) -> bool
        {
          return( std::strcmp( s, "--model" ) == 0 );
        }
        )
      );
  if( model < argc )
  {
    if( std::string( argv[ model ] ) == "linear" )
      return( linear_fit( argc, argv ) );
    else if( std::string( argv[ model ] ) == "logistic" )
      return( logistic_fit( argc, argv ) );
    else
    {
      std::cerr
        << "Usage: " << argv[ 0 ]
        << " ... --model [linear/logistic] ..."
        << std::endl;
      return( EXIT_FAILURE );
    } // end if
  }
  else
  {
    std::cerr
      << "Usage: " << argv[ 0 ]
      << " ... --model [linear/logistic] ..."
      << std::endl;
    return( EXIT_FAILURE );
  } // end if

  /* TODO
     std::string model = "linear";
     std::string optimizer = "ADAM";
     std::string csv = "cvs_file_with_data";
     boost::program_options::options_description opt { "Options" };
     opt.add_options( )
     ( "help,h", "help message" )
     PROGRAM_OPTIONS_OPTION( model, "model" )
     PROGRAM_OPTIONS_OPTION( csv, "csv" )
     PROGRAM_OPTIONS_OPTION( optimizer, "optimizer" );

     boost::program_options::variables_map m;
     boost::program_options::store(
     boost::program_options::parse_command_line( argc, argv, opt ), m
     );
     boost::program_options::notify( m );
     if( m.count( "help" ) )
     {
     std::cerr << opt << std::endl;
     return( EXIT_FAILURE );
     } // end if

     if( model == "linear" )
     {
     if( !linear_fit( opt, csv, optimizer ) )
     return( EXIT_FAILURE );
     }
     else if( model == "logistic" )
     {
     if( !logistic_fit( opt, csv, optimizer ) )
     return( EXIT_FAILURE );
     }
     else
     {
     std::cerr << opt << std::endl;
     std::cerr << "******** POSSIBLE MODELS ********" << std::endl;
     std::cerr << "linear,logistic" << std::endl;
     std::cerr << "*********************************" << std::endl;
     return( EXIT_FAILURE );
     } // end if
  */

  return( EXIT_SUCCESS );
}

// -------------------------------------------------------------------------
/* TODO
*/

// -------------------------------------------------------------------------
/* TODO
   bool linear_fit(
   boost::program_options::options_description& opt,
   const std::string& csv,
   const std::string& optimizer
   )
   {
   using TModel = ivqML::Model::Regression::Linear< TScalar >;
   using TCost = ivqML::Cost::MeanSquareError< TModel >;
   return( choose_optimizer< TCost >( opt, csv, optimizer ) );
   }
*/

// -------------------------------------------------------------------------
/* TODO
   bool logistic_fit(
   boost::program_options::options_description& opt,
   const std::string& csv,
   const std::string& optimizer
   )
   {
   using TModel = ivqML::Model::Regression::Logistic< TScalar >;
   using TCost = ivqML::Cost::BinaryCrossEntropy< TModel >;
   return( choose_optimizer< TCost >( opt, csv, optimizer ) );
   }
*/

// eof - $RCSfile$
