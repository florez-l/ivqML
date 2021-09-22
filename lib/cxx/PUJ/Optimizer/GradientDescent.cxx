// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================

#include <PUJ/Optimizer/GradientDescent.h>
#include <limits>
#include <random>

// -------------------------------------------------------------------------
template< class _TScalar, class _TTraits >
PUJ::Optimizer::GradientDescent< _TScalar, _TTraits >::
GradientDescent( const TCost& c, unsigned long d, const Self::InitType& t )
{
  this->m_Cost = c;
  this->m_Dimensions = d;

  this->m_Alpha = TScalar( 1e-2 );
  this->m_Lambda = TScalar( 0 );
  this->m_Epsilon = std::numeric_limits< TScalar >::epsilon( );
  this->m_MaximumNumberOfIterations = 100000;
  this->m_DebugIterations = 1000;
  this->m_Debug =
    []( const TScalar&, const TScalar&, const TRow&, unsigned long long )
    {};

  if( t == Self::RandomInit )
  {
    std::random_device dev;
    std::mt19937 gen( dev( ) );
    std::uniform_real_distribution< TScalar > dist( -1, 1 );

    this->m_Theta =
      TRow::NullaryExpr(
        this->m_Dimensions,
        [&]( typename TRow::Index i ) -> TScalar
        {
          return( dist( gen ) );
        }
        );
  }
  else if( t == Self::ZerosInit )
    this->m_Theta = TRow::Zero( this->m_Dimensions );
  else if( t == Self::OnesInit )
    this->m_Theta = TRow::Ones( this->m_Dimensions );
}

// -------------------------------------------------------------------------
template< class _TScalar, class _TTraits >
const typename PUJ::Optimizer::GradientDescent< _TScalar, _TTraits >::
TCost& PUJ::Optimizer::GradientDescent< _TScalar, _TTraits >::
GetCost( ) const
{
  return( this->m_Cost );
}

// -------------------------------------------------------------------------
template< class _TScalar, class _TTraits >
const typename PUJ::Optimizer::GradientDescent< _TScalar, _TTraits >::
TRow& PUJ::Optimizer::GradientDescent< _TScalar, _TTraits >::
GetTheta( ) const
{
  return( this->m_Theta );
}

// -------------------------------------------------------------------------
template< class _TScalar, class _TTraits >
const typename PUJ::Optimizer::GradientDescent< _TScalar, _TTraits >::
TScalar& PUJ::Optimizer::GradientDescent< _TScalar, _TTraits >::
GetAlpha( ) const
{
  return( this->m_Alpha );
}

// -------------------------------------------------------------------------
template< class _TScalar, class _TTraits >
const typename PUJ::Optimizer::GradientDescent< _TScalar, _TTraits >::
TScalar& PUJ::Optimizer::GradientDescent< _TScalar, _TTraits >::
GetLambda( ) const
{
  return( this->m_Lambda );
}

// -------------------------------------------------------------------------
template< class _TScalar, class _TTraits >
const typename PUJ::Optimizer::GradientDescent< _TScalar, _TTraits >::
TScalar& PUJ::Optimizer::GradientDescent< _TScalar, _TTraits >::
GetEpsilon( ) const
{
  return( this->m_Epsilon );
}

// -------------------------------------------------------------------------
template< class _TScalar, class _TTraits >
const unsigned long long&
PUJ::Optimizer::GradientDescent< _TScalar, _TTraits >::
GetMaximumNumberOfIterations( ) const
{
  return( this->m_MaximumNumberOfIterations );
}

// -------------------------------------------------------------------------
template< class _TScalar, class _TTraits >
const unsigned long long&
PUJ::Optimizer::GradientDescent< _TScalar, _TTraits >::
GetDebugIterations( ) const
{
  return( this->m_DebugIterations );
}

// -------------------------------------------------------------------------
template< class _TScalar, class _TTraits >
void PUJ::Optimizer::GradientDescent< _TScalar, _TTraits >::
SetAlpha( const TScalar& a )
{
  this->m_Alpha = a;
}

// -------------------------------------------------------------------------
template< class _TScalar, class _TTraits >
void PUJ::Optimizer::GradientDescent< _TScalar, _TTraits >::
SetLambda( const TScalar& l )
{
  this->m_Lambda = l;
}

// -------------------------------------------------------------------------
template< class _TScalar, class _TTraits >
void PUJ::Optimizer::GradientDescent< _TScalar, _TTraits >::
SetEpsilon( const TScalar& e )
{
  this->m_Epsilon = e;
}

// -------------------------------------------------------------------------
template< class _TScalar, class _TTraits >
void PUJ::Optimizer::GradientDescent< _TScalar, _TTraits >::
SetMaximumNumberOfIterations( const unsigned long long& i )
{
  this->m_MaximumNumberOfIterations = i;
}

// -------------------------------------------------------------------------
template< class _TScalar, class _TTraits >
void PUJ::Optimizer::GradientDescent< _TScalar, _TTraits >::
SetDebugIterations( const unsigned long long& i )
{
  this->m_DebugIterations = i;
}

// -------------------------------------------------------------------------
template< class _TScalar, class _TTraits >
void PUJ::Optimizer::GradientDescent< _TScalar, _TTraits >::
SetDebug( TDebug f )
{
  this->m_Debug = f;
}

// -------------------------------------------------------------------------
template< class _TScalar, class _TTraits >
void PUJ::Optimizer::GradientDescent< _TScalar, _TTraits >::
Fit( )
{
  TRow dt( this->m_Dimensions );
  TScalar Jn, J = this->m_Cost( this->m_Theta, &dt );
  TScalar dJ = std::numeric_limits< TScalar >::max( );
  bool stop = false;

  unsigned long long i = 0;
  this->m_Debug( J, dJ, this->m_Theta, i );
  while( !stop )
  {
    this->m_Theta -= dt * this->m_Alpha;

    Jn = this->m_Cost( this->m_Theta, &dt );
    dJ = J - Jn;
    J = Jn;

    i++;
    if( i % this->m_DebugIterations == 0 )
      this->m_Debug( J, dJ, this->m_Theta, i );
    stop = ( i > this->m_MaximumNumberOfIterations || dJ <= this->m_Epsilon );
  } // end while
  this->m_Debug( J, dJ, this->m_Theta, i );
}

// -------------------------------------------------------------------------
#include <PUJ_ML_export.h>
template class PUJ_ML_EXPORT PUJ::Optimizer::GradientDescent< float >;
template class PUJ_ML_EXPORT PUJ::Optimizer::GradientDescent< double >;
template class PUJ_ML_EXPORT PUJ::Optimizer::GradientDescent< long double >;

// eof - $RCSfile$
