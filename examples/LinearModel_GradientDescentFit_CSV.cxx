// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================

#include <ivqML/IO/CSV.h>
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
  ivqMLAttributeMacro( input, std::string, "" );

public:
  Training( )
    : Superclass( )
    {
      this->m_P.add_options( )
        ivqML_Optimizer_OptionMacro( input, "input" );
    }
  virtual ~Training( ) override = default;

protected:
  virtual void _prepare_training( ) override
    {
      // Data
      TMatrix D;
      ivqML::IO::CSV::Read( D, this->m_input );
      this->m_dX = D.block( 0, 0, D.rows( ), D.cols( ) - 1 );
      this->m_dY = D.col( D.cols( ) - 1 );

      // Model to be fitted
      this->m_Model.set_number_of_inputs( this->m_dX.cols( ) );
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
