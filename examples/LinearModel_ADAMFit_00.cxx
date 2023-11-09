// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================

#include <csignal>
#include <iostream>

#include <ivqML/Model/Linear.h>
#include <ivqML/Cost/MSE.h>
#include <ivqML/Optimizer/ADAM.h>

using _R = long double;
using _M = ivqML::Model::Linear< _R >;
using _C = ivqML::Cost::MSE< _M >;

/**
 */
class Training
  : public ivqML::Optimizer::ADAM< _C >
{
public:
  using Self = Training;
  using Superclass = ivqML::Optimizer::ADAM< _C >;
  ivqML_Optimizer_Typedefs;

public:
  ivqMLAttributeMacro( samples, TNatural, 100 );

public:
  Training( );
  virtual ~Training( ) override = default;

  static bool debug(
    const _R& J, const _R& G, const _M* m, const _M::TNatural& i, bool d
    );

  virtual void fit( ) override;

protected:
  _M m_FittedModel;
  _M::TMatrix m_dX;
  _M::TMatrix m_dY;
  static bool s_ManualStop;
};
bool Training::s_ManualStop = false;

// -------------------------------------------------------------------------
int main( int argc, char** argv )
{
  Training tr_exp;

  std::string ret = tr_exp.parse_options( argc, argv );
  if( ret != "" )
  {
    std::cerr << ret << std::endl;
    return( EXIT_FAILURE );
  } // end if

  tr_exp.fit( );

  return( EXIT_SUCCESS );
}

// -------------------------------------------------------------------------
Training::
Training( )
  : Superclass( )
{
  this->m_P.add_options( )
    ivqML_Optimizer_OptionMacro( samples, "samples" );

  // Detect ctrl-c event to stop optimization and finish training
  signal( SIGINT, []( int s ) -> void { Self::s_ManualStop = true; } );

  // Some basic configuration
  this->set_debug( Self::debug );
}

// -------------------------------------------------------------------------
bool Training::
debug( const _R& J, const _R& G, const _M* m, const _M::TNatural& i, bool d )
{
  if( d )
    std::cout << "J=" << J << ", Gn=" << G << ", i=" << i << std::endl;
  return( Self::s_ManualStop );
}

// -------------------------------------------------------------------------
void Training::
fit( )
{
  // Model to generate data
  _M real_model( 1 );
  real_model[ 0 ] = 3;
  real_model[ 1 ] = -2.5;
  std::cout << "Real model    : " << real_model << std::endl;

  // Some random input data
  this->m_dX =
    _M::TMatrix::Zero( this->m_samples, real_model.number_of_inputs( ) );
  this->m_dX.setRandom( );
  this->m_dX.array( ) *= 10;
  this->m_dX.array( ) -= 5;

  this->m_dY = _M::TMatrix::Zero( this->m_dX.rows( ), 1 );
  real_model( this->m_dY, this->m_dX );

  // Model to be fitted
  this->m_FittedModel
    .set_number_of_parameters( real_model.number_of_parameters( ) );
  this->m_FittedModel.random_fill( );
  std::cout << "Initial model : " << this->m_FittedModel << std::endl;

  // Go!
  this->init( this->m_FittedModel, this->m_dX, this->m_dY );
  this->Superclass::fit( );
  std::cout << "Fitted model  : " << this->m_FittedModel << std::endl;
}

// eof - $RCSfile$
