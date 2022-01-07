#include <algorithm>
#include <iostream>
#include <random>
#include <sstream>
#include <PUJ_ML/Model/Linear.h>

using TScalar = long double;
using TModel = PUJ_ML::Model::Linear< TScalar >;

int main( int argc, char** argv )
{
  TScalar slope = 1;
  TScalar offset = 0;
  TScalar min_x = -100;
  TScalar max_x =  100;
  unsigned long long n_samples = 100;
  
  TScalar learning_rate = 1e-1;
  TScalar epsilon = 1e-8;
  unsigned long long n_iter = 10000;
  unsigned long long d_iter = 100;
  
  // Create data
  auto X = TModel::TCol::LinSpaced( n_samples, min_x, max_x );
  TModel::TCol Y = ( X.array( ) * slope ) + offset;
  
  // Permute data
  Eigen::PermutationMatrix< Eigen::Dynamic > P( X.rows( ) );
  P.setIdentity( );
  std::random_device rnd_dev;
  std::mt19937 rnd_eng( rnd_dev );
  std::random_shuffle( P.indices( ).data( ), P.indices( ).data( ) + P.indices( ).size( ), rnd_eng );
  
  //X = P * X;
  //Y = P * Y;
  
  std::cout << Y << std::endl;
    
  /* TODO
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
  */
  return( EXIT_SUCCESS );
}

// eof - $RCSfile$
