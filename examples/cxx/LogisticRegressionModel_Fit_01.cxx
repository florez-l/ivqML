// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================

#include <PUJ_ML/IO/CSV.h>
#include <PUJ_ML/Model/Regression/Logistic.h>
#include <PUJ_ML/Optimizers/GradientDescent.h>

int main( int argc, char** argv )
{
  // Model types
  using TReal = long double;
  using TModel = PUJ_ML::Model::Regression::Logistic< TReal >;
  using TMatrix = TModel::TMatrix;
  using TCol = TModel::TCol;
  using TRow = TModel::TRow;

  // Read CSV data
  TMatrix D;
  PUJ_ML::IO::CSV::read( D, argv[ 1 ] );
  auto X = D.block( 0, 0, D.rows( ), D.cols( ) - 1 );
  auto Y = D.block( 0, D.cols( ) - 1, D.rows( ), 1 );

  // Optimization types
  using TOptimizer =
    PUJ_ML::Optimizers::GradientDescent
    < TModel::Cost, decltype( X ), decltype( Y ) >;

  // Configure optimization
  TModel model;
  TOptimizer opt( model, X, Y );
  opt.set_learning_rate( 1e-6 );
  opt.set_batch_size( 0 );
  opt.set_regularization_coefficient( 0 );
  opt.set_regularization_type_to_ridge( );
  // TODO: opt.set_regularization_type_to_LASSO( );
  // TODO: opt.set_gradient_epsilon( 1e-8 );
  // TODO: opt.set_maximum_epochs( 10000 );
  opt.set_debugging_epochs( 100 );
  opt.set_debug(
    [](
      const TReal& J,
      const TReal& nG,
      const unsigned long long& epoch
      ) -> bool
    {
      std::cout << J << " " << nG << " " << epoch << std::endl;
      return( false );
    }
    );
  opt.fit( );

  std::cout << "===============" << std::endl;
  std::cout << model << std::endl;
  std::cout << "===============" << std::endl;

  return( EXIT_SUCCESS );
}

// eof - $RCSfile$