// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================

#include <iostream>
#include <random>
#include <PUJ/Model.h>

// -- Typedefs
using TModel = PUJ::Model::Logistic< double >;

int main( int argc, char** argv )
{
  TModel::TRowVector w( 1 );
  w << 2;
  TModel::TScalar b = -1;

  TModel m( w, b );

  TModel::TRowVector s( 1 );
  s << 3;
  std::cout << "----------------------" << std::endl;
  std::cout << m( s ) << std::endl;
  std::cout << "----------------------" << std::endl;
  std::cout << m( s, false ) << std::endl;
  std::cout << "----------------------" << std::endl;

  std::random_device rd;
  std::mt19937 gen( rd( ) );
  std::uniform_real_distribution< TModel::TScalar > dis( -10.0, 10.0 );
  TModel::TMatrix X =
    TModel::TMatrix::NullaryExpr( 30, 1, [&](){ return( dis( gen ) ); } );
  std::cout << "----------------------" << std::endl;
  std::cout << m( X ) << std::endl;
  std::cout << "----------------------" << std::endl;
  std::cout << m( X, false ) << std::endl;
  std::cout << "----------------------" << std::endl;

  return( EXIT_SUCCESS );
}

// eof - $RCSfile$
