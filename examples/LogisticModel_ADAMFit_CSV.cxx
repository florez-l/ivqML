// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================

#include <csignal>
#include <iostream>

#include <ivqML/IO/CSV.h>
#include <ivqML/Model/Logistic.h>
#include <ivqML/Optimizer/ADAM.h>

using _R = long double;
using _M = ivqML::Model::Logistic< _R >;

/**
 */
class Training
  : public ivqML::Optimizer::ADAM< _M, Eigen::Block< _M::TMatrix >, _M::TMatrix::ColXpr >
{
public:
  using Self = Training;
  using Superclass = ivqML::Optimizer::ADAM< _M, Eigen::Block< _M::TMatrix >, _M::TMatrix::ColXpr >;
  ivqML_Optimizer_Typedefs;

public:
  ivqMLAttributeMacro( input, std::string, "" );

public:
  Training( );
  virtual ~Training( ) override = default;

  static bool debug(
    const _R& J, const _R& G, const _M* m, const _M::TNatural& i, bool d
    );

  virtual void fit( ) override;

protected:
  _M m_FittedModel;
  _M::TMatrix m_D;
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
    ivqML_Optimizer_OptionMacro( input, "input" );

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
  // Data
  ivqML::IO::CSV::Read( this->m_D, this->m_input );

  // Model to be fitted
  this->m_FittedModel.set_number_of_inputs( this->m_D.cols( ) - 1 );
  this->m_FittedModel.random_fill( );
  std::cout << "Initial model : " << this->m_FittedModel << std::endl;

  // Go!
  this->init(
    this->m_FittedModel,
    this->m_D.block( 0, 0, this->m_D.rows( ), this->m_D.cols( ) - 1 ),
    this->m_D.col( this->m_D.cols( ) - 1 )
    );
  this->Superclass::fit( );
  std::cout << "Fitted model  : " << this->m_FittedModel << std::endl;
}

// eof - $RCSfile$
