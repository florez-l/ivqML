// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __PUJ__NeuralNetwork__ActivationFunction__h__
#define __PUJ__NeuralNetwork__ActivationFunction__h__

#include <PUJ/Traits.h>

// -------------------------------------------------------------------------
#define PUJ_NN_ActivationFunction_Base( _n_ ) \
      template< class _TScalar, class _TTraits > \
      typename PUJ::Model::NeuralNetwork::_n_< _TScalar >::TMatrix \
      PUJ::Model::NeuralNetwork::_n_< _TScalar >:: \
      operator()( const TMatrix& z )

// -------------------------------------------------------------------------
PUJ_NN_ActivationFunction_Base( ArcTan )
{
}
      
// -------------------------------------------------------------------------
PUJ_NN_ActivationFunction_Base( BinaryStep )
{
  static const auto f = []( TScalar v ) -> TScalar
    {
      return( ( v <  TScalar( 0 ) )? TScalar( 0 ): TScalar( 1 ) );
    };
  return( z.unaryExpr( f ) );
}

// -------------------------------------------------------------------------
PUJ_NN_ActivationFunction_Base( ELU )
{
}

// -------------------------------------------------------------------------
PUJ_NN_ActivationFunction_Base( Identity )
{
  return( z );
}

// -------------------------------------------------------------------------
PUJ_NN_ActivationFunction_Base( LeakyReLU )
{
  static const auto f = []( TScalar v ) -> TScalar
    {
      return( ( v <  TScalar( 0 ) )? TScalar( 1e-2 ) * v: v );
    };
  return( z.unaryExpr( f ) );
}

// -------------------------------------------------------------------------
PUJ_NN_ActivationFunction_Base( Logistic )
{
  static const auto f = []( TScalar v ) -> TScalar
    {
      static const TScalar _0 = TScalar( 0 );
      static const TScalar _1 = TScalar( 1 );
      static const TScalar _bnd = TScalar( 40 );

      if     ( v >  _bnd ) return( _1 );
      else if( v < -_bnd ) return( _0 );
      else                 return( _1 / ( _1 + std::exp( -v ) ) );
    };
  return( z.unaryExpr( f ) );
}

// -------------------------------------------------------------------------
PUJ_NN_ActivationFunction_Base( RandomReLU )
{
  static const auto f = []( TScalar v ) -> TScalar
    {
      return( ( v <  TScalar( 0 ) )? this->m_Alpha * v: v );
    };
  return( z.unaryExpr( f ) );
}

// -------------------------------------------------------------------------
PUJ_NN_ActivationFunction_Base( ReLU )
{
  static const auto f = []( TScalar v ) -> TScalar
    {
      return( ( v <  TScalar( 0 ) )? TScalar( 0 ): v );
    };
  return( z.unaryExpr( f ) );
}

// -------------------------------------------------------------------------
PUJ_NN_ActivationFunction_Base( SoftMax )
{
}

// -------------------------------------------------------------------------
PUJ_NN_ActivationFunction_Base( SoftMin )
{
}

// -------------------------------------------------------------------------
PUJ_NN_ActivationFunction_Base( Tanh )
{
}

// eof - $RCSfile$
