// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================

#include <csignal>
#include <iostream>

#include "Helpers.h"

#include <itkImage.h>
#include <itkImageFileWriter.h>
#include <ivq/ITK/ImageFileReader.h>

#include <ivqML/Model/FeedForwardNetwork.h>
#include <ivqML/Optimizer/ADAM.h>
#include <ivqML/ITK/ApplyModelToImageMeshFilter.h>

using _R = double;
using _M = ivqML::Model::FeedForwardNetwork< _R >;
using _I = itk::Image< _R, 2 >;

/**
 */
class Training
  : public ivqML::Optimizer::ADAM< _M >
{
public:
  using Self = Training;
  using Superclass = ivqML::Optimizer::ADAM< _M >;
  ivqML_Optimizer_Typedefs;

public:
  ivqMLAttributeMacro( input, std::string, "" );
  ivqMLAttributeMacro( output, std::string, "" );
  ivqMLAttributeMacro( model, std::string, "" );
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
    ivqML_Optimizer_OptionMacro( samples, "samples" )
    ivqML_Optimizer_OptionMacro( input, "input" )
    ivqML_Optimizer_OptionMacro( output, "output" )
    ivqML_Optimizer_OptionMacro( model, "model" );

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
  // Get input data
  auto reader = ivq::ITK::ImageFileReader< _I >::New( );
  reader->SetFileName( this->m_input );
  reader->NormalizeOn( );
  reader->Update( );

  auto D =
    ivqML::Helpers::extract_discrete_samples_from_image< _I, _M::TMatrix >(
      reader->GetOutput( ), this->m_samples, 2
      );
  this->m_dX = D.block( 0, 0, D.rows( ), D.cols( ) - 1 );
  this->m_dY = D.block( 0, D.cols( ) - 1, D.rows( ), 1 );

  // Model to be fitted
  std::ifstream d_str( this->m_model.c_str( ) );
  d_str >> this->m_FittedModel;
  d_str.close( );
  this->m_FittedModel.init( );

  std::cout
    << "Initial model : "
    << std::endl << this->m_FittedModel << std::endl;

  // Go!
  this->init( this->m_FittedModel, this->m_dX, this->m_dY );
  this->Superclass::fit( );
  std::cout << "Fitted model  : " << this->m_FittedModel << std::endl;

  // Save result image
  using _A = ivqML::ITK::ApplyModelToImageMeshFilter< _I, _M >;
  auto apply_model = _A::New( );
  apply_model->SetInput( reader->GetOutput( ) );
  apply_model->SetModel( this->m_FittedModel );

  auto writer = itk::ImageFileWriter< _A::TOutput >::New( );
  writer->SetInput( apply_model->GetOutput( ) );
  writer->SetFileName( this->m_output );
  writer->Update( );
}

// eof - $RCSfile$
