// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================

#include <PUJ/Model/NeuralNetwork/ActivationFunction.h>

// -------------------------------------------------------------------------
#define PUJ_NN_ActivationFunction_Base( _n_ )                           \
  template< class _TScalar, class _TTraits >                            \
  typename PUJ::Model::NeuralNetwork::_n_< _TScalar, _TTraits >::TMatrix \
  PUJ::Model::NeuralNetwork::_n_< _TScalar, _TTraits >::                \
  operator()( const TMatrix& z )

// -------------------------------------------------------------------------
PUJ_NN_ActivationFunction_Base( ArcTan )
{
  static const auto f = []( TScalar v ) -> TScalar
    {
      return( std::atan( v ) );
    };
  return( z.unaryExpr( f ) );
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
  static const auto f = [=]( TScalar v ) -> TScalar
    {
      if( v < TScalar( 0 ) )
        return( this->m_Alpha * ( std::exp( v ) - TScalar( 1 ) ) );
      else
        return( v );
    };
  return( z.unaryExpr( f ) );
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
  static const auto f = [=]( TScalar v ) -> TScalar
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
  static const auto f = []( TScalar v ) -> TScalar
    {
      return( std::exp( v ) );
    };
  auto e =  z.unaryExpr( f );
  return( e / e.sum( ) );
}

// -------------------------------------------------------------------------
PUJ_NN_ActivationFunction_Base( SoftPlus )
{
  static const auto f = []( TScalar v ) -> TScalar
    {
      return( std::log( TScalar( 1 ) + std::exp( v ) ) );
    };
  return( z.unaryExpr( f ) );
}

// -------------------------------------------------------------------------
PUJ_NN_ActivationFunction_Base( Tanh )
{
  static const auto f = []( TScalar v ) -> TScalar
    {
      return( std::tanh( v ) );
    };
  return( z.unaryExpr( f ) );
}

// -------------------------------------------------------------------------
#define PUJ_NN_ActivationFunction_Derivative( _n_ )                     \
  template< class _TScalar, class _TTraits >                            \
  typename PUJ::Model::NeuralNetwork::_n_< _TScalar, _TTraits >::TMatrix \
  PUJ::Model::NeuralNetwork::_n_< _TScalar, _TTraits >::                \
  operator[]( const TMatrix& z )

// -------------------------------------------------------------------------
PUJ_NN_ActivationFunction_Derivative( ArcTan )
{
  static const auto f = []( TScalar v ) -> TScalar
    {
      return( TScalar( 1 ) / ( TScalar( 1 ) + ( v * v ) ) );
    };
  return( z.unaryExpr( f ) );
}

// -------------------------------------------------------------------------
PUJ_NN_ActivationFunction_Derivative( BinaryStep )
{
  return( TMatrix::Zero( z.rows( ), z.cols( ) ) );
}

// -------------------------------------------------------------------------
PUJ_NN_ActivationFunction_Derivative( ELU )
{
  static const auto f = [=]( TScalar v ) -> TScalar
    {
      if( v < TScalar( 0 ) )
        return( this->m_Alpha * std::exp( v ) );
      else
        return( TScalar( 1 ) );
    };
  return( z.unaryExpr( f ) );
}

// -------------------------------------------------------------------------
PUJ_NN_ActivationFunction_Derivative( Identity )
{
  return( TMatrix::Ones( z.rows( ), z.cols( ) ) );
}

// -------------------------------------------------------------------------
PUJ_NN_ActivationFunction_Derivative( LeakyReLU )
{
  static const auto f = []( TScalar v ) -> TScalar
    {
      return( ( v <  TScalar( 0 ) )? TScalar( 1e-2 ): TScalar( 1 ) );
    };
  return( z.unaryExpr( f ) );
}

// -------------------------------------------------------------------------
PUJ_NN_ActivationFunction_Derivative( Logistic )
{
  static const auto f = []( TScalar v ) -> TScalar
    {
      static const TScalar _0 = TScalar( 0 );
      static const TScalar _1 = TScalar( 1 );
      static const TScalar _bnd = TScalar( 40 );

      if( -_bnd < v && v < _bnd )
      {
        TScalar e = _1 / ( _1 + std::exp( -v ) );
        return( e * ( _1 - e ) );
      }
      else
        return( TScalar( 0 ) );
    };
  return( z.unaryExpr( f ) );
}

// -------------------------------------------------------------------------
PUJ_NN_ActivationFunction_Derivative( RandomReLU )
{
  static const auto f = [=]( TScalar v ) -> TScalar
    {
      return( ( v <  TScalar( 0 ) )? this->m_Alpha: TScalar( 1 ) );
    };
  return( z.unaryExpr( f ) );
}

// -------------------------------------------------------------------------
PUJ_NN_ActivationFunction_Derivative( ReLU )
{
  static const auto f = []( TScalar v ) -> TScalar
    {
      return( ( v <  TScalar( 0 ) )? TScalar( 0 ): TScalar( 1 ) );
    };
  return( z.unaryExpr( f ) );
}

// -------------------------------------------------------------------------
PUJ_NN_ActivationFunction_Derivative( SoftMax )
{
  /* TODO
     static const auto f = []( TScalar v ) -> TScalar
     {
     return( std::exp( v ) );
     };
     auto e =  z.unaryExpr( f );
     return( e / e.sum( ) );
  */
  return( TMatrix::Zero( z.rows( ), z.cols( ) ) );
}

// -------------------------------------------------------------------------
PUJ_NN_ActivationFunction_Derivative( SoftPlus )
{
  static const auto f = []( TScalar v ) -> TScalar
    {
      TScalar e = std::exp( v );
      return( e / ( e + TScalar( 1 ) ) );
    };
  return( z.unaryExpr( f ) );
}

// -------------------------------------------------------------------------
PUJ_NN_ActivationFunction_Derivative( Tanh )
{
  static const auto f = []( TScalar v ) -> TScalar
    {
      TScalar t = std::tanh( v );
      return( TScalar( 1 ) - ( t * t ) );
    };
  return( z.unaryExpr( f ) );
}

// -------------------------------------------------------------------------
#define PUJ_NN_ActivationFunction_Instances( _n_ )                      \
  template class PUJ_ML_EXPORT PUJ::Model::NeuralNetwork::_n_< float >; \
  template class PUJ_ML_EXPORT PUJ::Model::NeuralNetwork::_n_< double >

PUJ_NN_ActivationFunction_Instances( ArcTan );
PUJ_NN_ActivationFunction_Instances( BinaryStep );
PUJ_NN_ActivationFunction_Instances( ELU );
PUJ_NN_ActivationFunction_Instances( Identity );
PUJ_NN_ActivationFunction_Instances( LeakyReLU );
PUJ_NN_ActivationFunction_Instances( Logistic );
PUJ_NN_ActivationFunction_Instances( RandomReLU );
PUJ_NN_ActivationFunction_Instances( ReLU );
PUJ_NN_ActivationFunction_Instances( SoftMax );
PUJ_NN_ActivationFunction_Instances( SoftPlus );
PUJ_NN_ActivationFunction_Instances( Tanh );

// eof - $RCSfile$
