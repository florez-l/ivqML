#include <iostream>
#include <PUJ_ML/Model/Linear.h>

using TScalar = long double;
using TModel = PUJ_ML::Model::Linear< TScalar >;

int main( int argc, char** argv )
{
  TScalar p[] = { 0, 1 };

  TModel m;
  m.SetParameters( p, p + 2 );

  TModel::TMatrix x( 3, 1 );
  x << 3, -4, 5;

  std::cout << "---------------------" << std::endl;
  std::cout << m << std::endl;
  std::cout << "---------------------" << std::endl;
  std::cout << m( x ) << std::endl;
  std::cout << "---------------------" << std::endl;
  std::cout << m[ x ] << std::endl;
  std::cout << "---------------------" << std::endl;

  return( EXIT_SUCCESS );
}

// eof - $RCSfile$
