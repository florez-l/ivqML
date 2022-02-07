// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __PUJ_ML__Optimizer__GradientDescent__hxx__
#define __PUJ_ML__Optimizer__GradientDescent__hxx__

#include <limits>

// -------------------------------------------------------------------------
template< class _M >
PUJ_ML::Optimizer::GradientDescent< _M >::
GradientDescent( )
{
  this->m_E = std::numeric_limits< TScalar >::epsilon( );
  this->SetRegularizationToRidge( );
  this->UnsetDebug( );
}

// -------------------------------------------------------------------------
template< class _M >
typename PUJ_ML::Optimizer::GradientDescent< _M >::
TCost* PUJ_ML::Optimizer::GradientDescent< _M >::
GetCost( ) const
{
  return( this->m_Cost );
}

// -------------------------------------------------------------------------
template< class _M >
void PUJ_ML::Optimizer::GradientDescent< _M >::
SetCost( TCost& c )
{
  this->m_Cost = &c;
}

// -------------------------------------------------------------------------
template< class _M >
const typename PUJ_ML::Optimizer::GradientDescent< _M >::
TScalar& PUJ_ML::Optimizer::GradientDescent< _M >::
GetLearningRate( ) const
{
  return( this->m_A );
}

// -------------------------------------------------------------------------
template< class _M >
void PUJ_ML::Optimizer::GradientDescent< _M >::
SetLearningRate( const TScalar& a )
{
  this->m_A = a;
}

// -------------------------------------------------------------------------
template< class _M >
const typename PUJ_ML::Optimizer::GradientDescent< _M >::
TScalar& PUJ_ML::Optimizer::GradientDescent< _M >::
GetRegularizationCoefficient( ) const
{
  return( this->m_L );
}

// -------------------------------------------------------------------------
template< class _M >
void PUJ_ML::Optimizer::GradientDescent< _M >::
SetRegularizationCoefficient( const TScalar& l )
{
  this->m_L = l;
}

// -------------------------------------------------------------------------
template< class _M >
const typename PUJ_ML::Optimizer::GradientDescent< _M >::
ERegularization& PUJ_ML::Optimizer::GradientDescent< _M >::
GetRegularization( ) const
{
  return( this->m_LType );
}

// -------------------------------------------------------------------------
template< class _M >
void PUJ_ML::Optimizer::GradientDescent< _M >::
SetRegularizationToRidge( )
{
  this->m_LType = Self::RidgeReg;
}

// -------------------------------------------------------------------------
template< class _M >
void PUJ_ML::Optimizer::GradientDescent< _M >::
SetRegularizationToLASSO( )
{
  this->m_LType = Self::LASSOReg;
}

// -------------------------------------------------------------------------
template< class _M >
const typename PUJ_ML::Optimizer::GradientDescent< _M >::
TScalar& PUJ_ML::Optimizer::GradientDescent< _M >::
GetEpsilon( ) const
{
  return( this->m_E );
}

// -------------------------------------------------------------------------
template< class _M >
void PUJ_ML::Optimizer::GradientDescent< _M >::
SetEpsilon( const TScalar& e )
{
  this->m_E = e;
}

// -------------------------------------------------------------------------
template< class _M >
const unsigned long long& PUJ_ML::Optimizer::GradientDescent< _M >::
GetNumberOfEpochs( ) const
{
  return( this->m_N );
}

// -------------------------------------------------------------------------
template< class _M >
void PUJ_ML::Optimizer::GradientDescent< _M >::
SetNumberOfEpochs( const unsigned long long& n )
{
  this->m_N = n;
}

// -------------------------------------------------------------------------
template< class _M >
const unsigned long long& PUJ_ML::Optimizer::GradientDescent< _M >::
GetDebugStep( ) const
{
  return( this->m_D );
}

// -------------------------------------------------------------------------
template< class _M >
void PUJ_ML::Optimizer::GradientDescent< _M >::
SetDebugStep( const unsigned long long& d )
{
  this->m_D = d;
}

// -------------------------------------------------------------------------
template< class _M >
void PUJ_ML::Optimizer::GradientDescent< _M >::
UnsetDebug( )
{
  this->m_Debug =
    []( unsigned long long i, TScalar J, bool show ) -> bool
    {
      return( false );
    };
}

// -------------------------------------------------------------------------
template< class _M >
void PUJ_ML::Optimizer::GradientDescent< _M >::
SetDebug( TDebug d )
{
  this->m_Debug = d;
}

// -------------------------------------------------------------------------
template< class _M >
void PUJ_ML::Optimizer::GradientDescent< _M >::
Fit( )
{
  if( this->m_Cost == nullptr )
    return;

  TCol g;
  TScalar Jp = std::numeric_limits< TScalar >::max( ), dJ;
  unsigned long long i = 0;
  bool stop;
  do
  {
    TScalar J = this->m_Cost->Compute( &g );
    if( this->m_L != TScalar( 0 ) )
    {
      const TCol& P = this->m_Cost->GetParameters( );
      if( this->m_LType == Self::RidgeReg )
      {
        J += P.array( ).pow( 2 ).sum( ) * this->m_L;
        g += P * TScalar( 2 ) * this->m_L;
      }
      else // if( this->m_LType == Self::LASSOReg )
      {
        J += P.array( ).abs( ).sum( ) * this->m_L;
        g +=
          TCol(
            ( ( P.array( ) > TScalar( 0 ) ) - ( P.array( ) < TScalar( 0 ) ) ).
            template cast< TScalar >( )
            ) * this->m_L;
      } // end if
    } // end if

    // Update parameters
    this->m_Cost->Update( g * this->m_A );

    // Next iteration
    stop = this->m_Debug( i, J, i % this->m_D == 0 );
    dJ = Jp - J;
    Jp = J;
    i++;
  } while( dJ > this->m_E && i <= this->m_N && !stop );
}

#endif // __PUJ_ML__Optimizer__GradientDescent__hxx__

// eof - $RCSfile$
