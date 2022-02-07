// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================

#include <PUJ_ML/Model/Perceptron.h>

// -------------------------------------------------------------------------
template< class _T >
PUJ_ML::Model::Perceptron< _T >::
Perceptron( )
  : Superclass( )
{
}

// -------------------------------------------------------------------------
template< class _T >
PUJ_ML::Model::Perceptron< _T >::
Perceptron( const TMatrix& X, const TCol& Y )
  : Superclass( )
{
}

// -------------------------------------------------------------------------
template< class _T >
typename PUJ_ML::Model::Perceptron< _T >::
TMatrix PUJ_ML::Model::Perceptron< _T >::
operator[]( const TMatrix& x )
{
  return(
    ( this->operator()( x ).array( ) > _T( 0 ) ).template cast< _T >( )
    );
}

// -------------------------------------------------------------------------
template class PUJ_ML_EXPORT PUJ_ML::Model::Perceptron< float >;
template class PUJ_ML_EXPORT PUJ_ML::Model::Perceptron< double >;
template class PUJ_ML_EXPORT PUJ_ML::Model::Perceptron< long double >;

// eof - $RCSfile$
