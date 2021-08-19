// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================

#include <PUJ/Model.h>

#include <iostream>


// -------------------------------------------------------------------------
template< class _TScalar >
PUJ::Model::Logistic< _TScalar >::
Logistic( const TRowVector& w, const TScalar& b )
  : Superclass( w, b )
{
}

// -------------------------------------------------------------------------
template< class _TScalar >
typename PUJ::Model::Logistic< _TScalar >::
TMatrix PUJ::Model::Logistic< _TScalar >::
operator()( const TMatrix& x, bool threshold ) const
{
  static const TScalar _1 = TScalar( 1 );
  static const TScalar _t = TScalar( 0.5 );

  auto z = _1 / ( _1 + Eigen::exp( -this->Superclass::operator()( x ).array( ) ) );
  if( threshold )
    return( ( z.array( ) >= _t ).template cast< TScalar >( ) );
  else
    return( z );
}

// -------------------------------------------------------------------------
#include <PUJ_ML_export.h>
template class PUJ_ML_EXPORT PUJ::Model::Logistic< float >;
template class PUJ_ML_EXPORT PUJ::Model::Logistic< double >;
template class PUJ_ML_EXPORT PUJ::Model::Logistic< long double >;

// eof - $RCSfile$
