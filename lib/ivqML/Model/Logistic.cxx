// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================

#include <ivqML/Model/Logistic.h>

// -------------------------------------------------------------------------
template< class _S >
ivqML::Model::Logistic< _S >::
Logistic( const TNatural& n )
  : Superclass( n )
{
}

// -------------------------------------------------------------------------
template< class _S >
void ivqML::Model::Logistic< _S >::
_evaluate( const TNatural& m ) const
{
  static const TScalar _0 = TScalar( 0 );
  static const TScalar _1 = TScalar( 1 );
  static const TScalar _E = std::numeric_limits< TScalar >::epsilon( );
  static const TScalar _L = std::log( _1 - _E ) - std::log( _E );
  static const auto f = [&]( TScalar z ) -> TScalar
    {
      if     ( z >  _L ) return( _1 );
      else if( z < -_L ) return( _0 );
      else               return( _1 / ( _1 + std::exp( -z ) ) );
    };

  this->Superclass::_evaluate( m );
  auto y = this->m_Y.block( 0, 0, m, this->m_Y.cols( ) );
  y.noalias( ) = y.unaryExpr( f );
}

// -------------------------------------------------------------------------
template class ivqML_EXPORT ivqML::Model::Logistic< float >;
template class ivqML_EXPORT ivqML::Model::Logistic< double >;
template class ivqML_EXPORT ivqML::Model::Logistic< long double >;

// eof - $RCSfile$
