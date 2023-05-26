// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================

#include <map>

#include <PUJ_ML/IO/CSV.h>
#include <PUJ_ML/Model/NeuralNetwork/FeedForward.h>
#include <PUJ_ML/Optimizers/ADAM.h>

int main( int argc, char** argv )
{
  // Model types
  using TReal = long double;
  using TModel = PUJ_ML::Model::NeuralNetwork::FeedForward< TReal >;
  using TMatrix = TModel::TMatrix;
  using TCol = TModel::TCol;
  using TRow = TModel::TRow;

  // Read CSV data
  TMatrix X, L;
  PUJ_ML::IO::CSV::read( X, argv[ 1 ] );
  PUJ_ML::IO::CSV::read( L, argv[ 2 ] );

  // Transform labels to bits
  std::map< TReal, unsigned long long > unique_labels;
  for( unsigned long long i = 0; i < L.size( ); ++i )
    unique_labels.insert(
      std::make_pair( L.data( )[ i ], unique_labels.size( )
        )
      );
  TMatrix Y = TMatrix::Zero( L.rows( ), unique_labels.size( ) );
  for( unsigned long long r = 0; r < L.rows( ); ++r )
  {
    auto uIt = unique_labels.find( L( r, 0 ) );
    Y( r, uIt->second ) = 1;
  } // end for

  // Initial model
  TModel model;
  std::ifstream in( argv[ 3 ] );
  std::istringstream in_str(
    std::string( std::istreambuf_iterator< char >( in ), { } )
    );
  in.close( );
  in_str >> model;

  // Optimization types
  using TOptimizer =
    PUJ_ML::Optimizers::ADAM
    < TModel::Cost, decltype( X ), decltype( Y ) >;

  // Configure optimization
  TOptimizer opt( model, X, Y );
  // TODO: opt.set_learning_rate( 1e-2 );
  opt.set_batch_size( 32 );
  opt.set_regularization_coefficient( 0 );
  opt.set_regularization_type_to_ridge( );
  // TODO: opt.set_regularization_type_to_LASSO( );
  // TODO: opt.set_gradient_epsilon( 1e-4 );
  opt.set_maximum_epochs( 10000 );
  opt.set_debugging_epochs( 1 );
  opt.set_debug(
    [](
      const TReal& J,
      const TReal& nG,
      const unsigned long long& epoch
      ) -> bool
    {
      std::cout
        << "Cost=" << J
        << " |Gradient|=" << nG
        << " epoch=" << epoch << std::endl;
      return( false );
    }
    );
  opt.fit( );

  /* TODO
     std::cout << "=== model ===" << std::endl;
     std::cout << model << std::endl;
     std::cout << "=============" << std::endl;

     TModel::TCol Z;
     model.threshold( Z, X );

     TModel::TMatrix lZ( Z.rows( ), 2 );
     lZ << TReal( 1 ) - Z.array( ), Z;

     TModel::TMatrix K = Y.transpose( ) * lZ;
     std::cout << "************************" << std::endl;
     std::cout << "*** Confusion matrix ***" << std::endl;
     std::cout << "************************" << std::endl;
     std::cout << K << std::endl;
     std::cout << "************************" << std::endl;
     std::cout << "Accuracy = " << K.diagonal( ).sum( ) / K.sum( ) << std::endl;
     std::cout << "************************" << std::endl;
  */

  return( EXIT_SUCCESS );
}

// eof - $RCSfile$
