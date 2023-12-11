// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================

#include <ivqML/Model/Linear.h>
#include <ivqML/Optimizer/GradientDescent.h>
#include <ivqML/Trainers/CommandLine.h>

using _R = long double;
using _M = ivqML::Model::Linear< _R >;
using _O = ivqML::Optimizer::GradientDescent< _M >;

/**
 */
class Training
  : public ivqML::Trainers::CommandLine< _O >
{
public:
  using Self = Training;
  using Superclass = ivqML::Trainers::CommandLine< _O >;
  ivqML_Optimizer_Typedefs;

public:
  ivqMLAttributeMacro( samples, TNatural, 100 );

public:
  Training( )
    : Superclass( )
    {
      this->m_P.add_options( )
        ivqML_Optimizer_OptionMacro( samples, "samples" );
    }
  virtual ~Training( ) override = default;

protected:
  virtual void _prepare_training( ) override
    {
      // Model to generate data
      _M real_model( 1 );
      real_model[ 0 ] = 3;
      real_model[ 1 ] = -2.5;
      std::cerr << "Real model: " << std::endl << real_model << std::endl;

      // Some random input data
      this->m_dX =
        _M::TMatrix::Zero( this->m_samples, real_model.number_of_inputs( ) );
      this->m_dX.setRandom( );
      this->m_dX.array( ) *= 10;
      this->m_dX.array( ) -= 5;
      this->m_dY = real_model.evaluate( this->m_dX );

      // Prepare model
      this->m_Model.set_number_of_parameters(
        real_model.number_of_parameters( )
        );
      this->m_Model.random_fill( );
    }
};

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

// eof - $RCSfile$
