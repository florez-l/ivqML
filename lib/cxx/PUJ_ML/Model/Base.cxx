// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================

#include <PUJ_ML/Model/Base.h>

// -------------------------------------------------------------------------
template< class _T >
PUJ_ML::Model::Base< _T >::Cost::
Cost( Self* model, const TMatrix& X, const TMatrix& Y )
{
  this->m_Model = model;
  this->m_X = &X;
  this->m_Y = &Y;
  this->m_Lambda = _T( 0 );
  this->set_ridge_regularization( );
}

// -------------------------------------------------------------------------
template< class _T >
const typename PUJ_ML::Model::Base< _T >::Cost::
ERegularization& PUJ_ML::Model::Base< _T >::Cost::
regularization( ) const
{
  return( this->m_Regularization );
}

// -------------------------------------------------------------------------
template< class _T >
void PUJ_ML::Model::Base< _T >::Cost::
set_ridge_regularization( )
{
  this->m_Regularization = Self::Cost::RidgeReg;
}

// -------------------------------------------------------------------------
template< class _T >
void PUJ_ML::Model::Base< _T >::Cost::
set_LASSO_regularization( )
{
  this->m_Regularization = Self::Cost::LASSOReg;
}

// -------------------------------------------------------------------------
template< class _T >
const _T& PUJ_ML::Model::Base< _T >::Cost::
lambda( ) const
{
  return( this->m_Lambda );
}

// -------------------------------------------------------------------------
template< class _T >
void PUJ_ML::Model::Base< _T >::Cost::
set_lambda( const _T& l )
{
  this->m_Lambda = l;

}

// -------------------------------------------------------------------------
template< class _T >
_T PUJ_ML::Model::Base< _T >::Cost::
operator()( _T* g ) const
{
  /*
   * TODO: take regularization into account
   */
  return( _T( 0 ) );
}

// -------------------------------------------------------------------------
template< class _T >
PUJ_ML::Model::Base< _T >::
Base( )
{
  this->m_P = TRow::Zero( 1 );
}

// -------------------------------------------------------------------------
template< class _T >
typename PUJ_ML::Model::Base< _T >::
TRow& PUJ_ML::Model::Base< _T >::
parameters( )
{
  return( this->m_P );
}

// -------------------------------------------------------------------------
template< class _T >
const typename PUJ_ML::Model::Base< _T >::
TRow& PUJ_ML::Model::Base< _T >::
parameters( ) const
{
  return( this->m_P );
}

// -------------------------------------------------------------------------
template< class _T >
unsigned long PUJ_ML::Model::Base< _T >::
number_of_parameters( ) const
{
  return( this->m_P.cols( ) );
}

// -------------------------------------------------------------------------
template< class _T >
void PUJ_ML::Model::Base< _T >::
set_parameters( const TRow& p )
{
  this->m_P = p;
}

// -------------------------------------------------------------------------
template< class _T >
typename PUJ_ML::Model::Base< _T >::
TColumn PUJ_ML::Model::Base< _T >::
operator[]( const TMatrix& x )
{
  return( this->operator()( x ) );
}

// -------------------------------------------------------------------------
template< class _T >
void PUJ_ML::Model::Base< _T >::
_Out( std::ostream& o ) const
{
  o << this->m_P.cols( ) << " " << this->m_P;
}

// -------------------------------------------------------------------------
template< class _T >
void PUJ_ML::Model::Base< _T >::
_In( std::istream& i )
{
  unsigned long long n;
  i >> n;
  this->m_P.resize( n );
  for( unsigned long long k = 0; k < n; ++k )
    i >> this->m_P( k );
}

// -------------------------------------------------------------------------
template class PUJ_ML::Model::Base< float >;
template class PUJ_ML::Model::Base< double >;
template class PUJ_ML::Model::Base< long double >;

// eof - $RCSfile$
