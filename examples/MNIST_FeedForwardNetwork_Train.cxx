
// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================

#include <ivqML/Model/FeedForwardNetwork.h>
#include <ivqML/Optimizer/ADAM.h>
#include <ivqML/Trainers/CommandLine.h>

using _R = long double;
using _M = ivqML::Model::FeedForwardNetwork< _R >;
using _O = ivqML::Optimizer::ADAM< _M >;

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
  ivqMLAttributeMacro( trainX, std::string, "" );
  ivqMLAttributeMacro( testX, std::string, "" );
  ivqMLAttributeMacro( trainY, std::string, "" );
  ivqMLAttributeMacro( testY, std::string, "" );

public:
  Training( )
    : Superclass( )
    {
      this->m_P.add_options( )
        ivqML_Optimizer_OptionMacro( trainX, "trainX" )
        ivqML_Optimizer_OptionMacro( testX, "testX" )
        ivqML_Optimizer_OptionMacro( trainY, "trainY" )
        ivqML_Optimizer_OptionMacro( testY, "testY" );
    }
  virtual ~Training( ) override = default;

protected:
  virtual void _prepare_training( ) override
    {
      // Data
      /* TODO
         TMatrix D;
         ivqML::IO::CSV::Read( D, this->m_input, 1 );
         this->m_dX = D.block( 0, 0, D.rows( ), D.cols( ) - 1 );
         this->m_dY = D.col( D.cols( ) - 1 );

         // Model to be fitted
         this->m_Model.set_number_of_inputs( this->m_dX.cols( ) );
         this->m_Model.random_fill( );
      */
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
