/* TODO
   #include <algorithm>
   #include <random>
   #include <sstream>
*/
#include <iostream>
#include <PUJ_ML/Helpers/CSV.h>
#include <PUJ_ML/Model/Linear.h>
#include <PUJ_ML/Optimizer/GradientDescent.h>


using TScalar = long double;
using TModel = PUJ_ML::Model::Linear< TScalar >;
using TMatrix = TModel::TMatrix;

int main( int argc, char** argv )
{
  std::string filename = argv[ 1 ];

  auto D = PUJ_ML::Helpers::CSV::Read< TMatrix >( filename, true );
  //TMatrix X = D.block( 0, 0, D.rows( ), 1 );
  //TMatrix Y = D.block( 0, 1, D.rows( ), 1 );

  TModel model;

  TModel::Cost J( &model, D.block( 0, 0, D.rows( ), D.cols( ) - 1 ), D.block( 0, D.cols( ) - 1, D.rows( ), 1 ) );

  PUJ_ML::Optimizer::GradientDescent< TModel > opt;
  opt.SetCost( J );
  opt.SetLearningRate( 1e-2 );
  opt.SetRegularizationCoefficient( 0 );
  opt.SetRegularizationToRidge( );
  // TODO: opt.SetRegularizationToLASSO( );
  opt.SetEpsilon( 1e-12 );
  opt.SetNumberOfEpochs( 10000 );
  opt.SetDebugStep( 100 );
  opt.Fit( );

  /* TODO
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
     std::random_shuffle(
     P.indices( ).data( ),
     P.indices( ).data( ) + P.indices( ).size( ),
     rnd_eng
     );
  
     //X = P * X;
     //Y = P * Y;
  
     std::cout << Y << std::endl;
  */

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
