// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __PUJ__Optimizer__GradientDescent__hxx__
#define __PUJ__Optimizer__GradientDescent__hxx__

#include <boost/program_options.hpp>

// -------------------------------------------------------------------------
template< class _TModel >
PUJ::Optimizer::GradientDescent< _TModel >::
GradientDescent( TCost* cost )
  : m_Cost( cost )
{
  this->m_Debug =
    []( unsigned long long, TScalar, TScalar, bool ) -> bool
    { return( false ); };
}

// -------------------------------------------------------------------------
template< class _TModel >
bool PUJ::Optimizer::GradientDescent< _TModel >::
ParseArguments( int argc, char** argv )
{
  namespace _TPo = boost::program_options;

  // Declare the supported options.
  _TPo::options_description desc( "Optimizing options" );
  desc.add_options( )
    ( "help", "produce help message" )
    ( "alpha", _TPo::value< TScalar >( &this->m_Alpha )->default_value( this->m_Alpha ), "learning rate" )
    ( "lambda", _TPo::value< TScalar >( &this->m_Lambda )->default_value( this->m_Lambda ), "regularization" )
    ( "max_iter", _TPo::value< unsigned long long >( &this->m_MaximumNumberOfIterations )->default_value( this->m_MaximumNumberOfIterations ), "maximum iterations" )
    ( "deb_iter", _TPo::value< unsigned long long >( &this->m_DebugIterations )->default_value( this->m_DebugIterations ), "iterations for debug" )
    ;

  _TPo::variables_map vm;
  _TPo::store( _TPo::parse_command_line( argc, argv, desc ), vm );
  _TPo::notify( vm );

  if( vm.count( "help" ) )
  {
    std::cout << desc << "\n";
    return( false );
  } // end if

  /* TODO
     if (vm.count("compression"))
     {
     std::cout << "Compression level was set to "
     << vm["compression"].as<int>() << ".\n";
     }
     else
     {
     std::cout << "Compression level was not set.\n";
     }
  */

  return( true );
}

// -------------------------------------------------------------------------
template< class _TModel >
void PUJ::Optimizer::GradientDescent< _TModel >::
SetRegularizationTypeToRidge( )
{
  this->SetRegularizationType( Self::RidgeRegType );
}

// -------------------------------------------------------------------------
template< class _TModel >
void PUJ::Optimizer::GradientDescent< _TModel >::
SetRegularizationTypeToLASSO( )
{
  this->SetRegularizationType( Self::LASSORegType );
}

// -------------------------------------------------------------------------
template< class _TModel >
const unsigned long long& PUJ::Optimizer::GradientDescent< _TModel >::
GetIterations( ) const
{
  return( this->m_Iterations );
}

// -------------------------------------------------------------------------
template< class _TModel >
void PUJ::Optimizer::GradientDescent< _TModel >::
SetDebug( TDebug d )
{
  this->m_Debug = d;
}

// -------------------------------------------------------------------------
template< class _TModel >
void PUJ::Optimizer::GradientDescent< _TModel >::
Fit( )
{
  static const TScalar maxJ = std::numeric_limits< TScalar >::max( );
  TScalar J = maxJ, Jn, dJ;
  TRow g;
  bool stop = false;
  this->m_Iterations = 0;
  do
  {
    // Next iteration
    for( unsigned int b = 0; b < this->m_Cost->GetNumberOfBatches( ); ++b )
    {
      Jn = this->m_Cost->operator()( b, &g );
      this->_Regularize( Jn, g );
      this->m_Cost->operator-=( g * this->m_Alpha );
    } // end if

    // Update cost difference
    dJ = ( J != maxJ )? J - Jn: J;

    // Update stop condition
    stop  =
      ( dJ <= this->m_Epsilon ) |
      ( this->m_MaximumNumberOfIterations <= this->m_Iterations ) |
      this->m_Debug(
        this->m_Iterations, J, dJ,
        this->m_Iterations % this->m_DebugIterations == 0
        );

    // Ok, finished an iteration
    J = Jn;
    this->m_Iterations++;
  } while( !stop );

  // Finish iteration
  this->m_Debug( this->m_Iterations, J, dJ, true );
}

// -------------------------------------------------------------------------
template< class _TModel >
void PUJ::Optimizer::GradientDescent< _TModel >::
_Regularize( TScalar& J, TRow& g )
{
  if( this->m_Lambda != TScalar( 0 ) )
  {
    const TRow& t = this->m_Cost->GetParameters( );
    if( this->m_RegularizationType == Self::RidgeRegType )
    {
      J += t.squaredNorm( ) * this->m_Lambda;
      g += t * TScalar( 2 ) * this->m_Lambda;
    }
    else if( this->m_RegularizationType == Self::LASSORegType )
    {
      J += t.array( ).abs( ).sum( ) * this->m_Lambda;
      // TODO:
    } // end if
  } // end if
}

#endif // __PUJ__Optimizer__GradientDescent__hxx__

// eof - $RCSfile$
