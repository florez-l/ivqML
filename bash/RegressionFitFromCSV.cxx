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
#include <ivqML/Trainer/CommandLine.h>
#include <ivqML/IO/CSV.h>

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
  char** csv = std::find( argv, argv + argc, std::string( "--csv" ) );
  if( csv != argv + argc )
    csv++;
  if( csv != argv + argc )
  {
    typename _TOptimizer::TModel::TMat D;
    if( ivqML::IO::CSV::Read( D, *csv ) )
    {
      ivqML::Trainer::CommandLine< _TOptimizer > opt;
      opt.set_data(
        D.block( 0, 0, D.rows( ), D.cols( ) - 1 ).transpose( ),
        D.block( 0, D.cols( ) - 1, D.rows( ), 1 ).transpose( )
        );
      std::string parsing = opt.parse_arguments( argc, argv );
      if( parsing == "" )
      {
        opt.fit( );
        std::cout
          << "Fitted model:" << std::endl << opt.model( )
          << std::endl;
        return( true );
      }
      else
      {
        std::cerr << parsing << std::endl;
        return( false );
      } // end if
    }
    else
    {
      std::cerr << "Could not read \"" << *csv << "\"" << std::endl;
      return( false );
    } // end if
  }
  else
  {
    std::cerr
      << "Usage: " << argv[ 0 ]
      << " ... --csv [input_file_name.csv] ..."
      << std::endl;
    return( false );
  } // end if
}

// -------------------------------------------------------------------------
template< class _TCost >
bool choose_optimizer( int argc, char** argv )
{
  char** optimizer =
    std::find( argv, argv + argc, std::string( "--optimizer" ) );
  if( optimizer != argv + argc )
  {
    optimizer++;
    if( optimizer != argv + argc )
    {
      if( *optimizer == std::string( "gd" ) )
        return(
          fit< ivqML::Optimizer::GradientDescent< _TCost > >( argc, argv )
          );
      else if( *optimizer == std::string( "ADAM" ) )
        return( fit< ivqML::Optimizer::ADAM< _TCost > >( argc, argv ) );
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
  char** model = std::find( argv, argv + argc, std::string( "--model" ) );
  if( model != argv + argc )
    model++;
  if( model != argv + argc )
  {
    if( *model == std::string( "linear" ) )
    {
      if( linear_fit( argc, argv ) )
        return( EXIT_SUCCESS );
      else
        return( EXIT_FAILURE );
    }
    else if( *model == std::string( "logistic" ) )
    {
      if( logistic_fit( argc, argv ) )
        return( EXIT_SUCCESS );
      else
        return( EXIT_FAILURE );
    }
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
}

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
