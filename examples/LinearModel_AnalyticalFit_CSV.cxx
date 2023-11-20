// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================

#include <iostream>
#include <boost/program_options.hpp>

using TScalar = long double;

// -------------------------------------------------------------------------
#define _Arg( _I, _V, _H )                                              \
  ( _I,                                                                 \
    boost::program_options::value< decltype( _V ) >( &_V )              \
    ->default_value( _V ), _H )

int main( int argc, char** argv )
{
  std::string input = "";
  TScalar lambda = 0;

  boost::program_options::options_description options { "Options." };
  options.add_options( )
    ( "help,h", "help message" )
    _Arg( "input,i",  input, "Input file" )
    _Arg( "lambda,l", lambda, "Regularization coefficient" );

  boost::program_options::variables_map vmap;
  boost::program_options::store(
    boost::program_options::parse_command_line( argc, argv, options ), vmap
    );
  boost::program_options::notify( vmap );
  if( vmap.count( "help" ) )
  {
    std::cerr << options << std::endl;
    return( EXIT_FAILURE );
  } // end if



  return( EXIT_SUCCESS );
}

// eof - $RCSfile$
