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
}

// -------------------------------------------------------------------------
template< class _T >
const typename PUJ_ML::Model::Base< _T >::
TCol& PUJ_ML::Model::Base< _T >::Cost::
GetParameters( ) const
{
  return( this->m_Model->GetParameters( ) );
}

// -------------------------------------------------------------------------
template< class _T >
void PUJ_ML::Model::Base< _T >::Cost::
Update( const TCol& g ) const
{
  if( this->m_Model != nullptr )
    this->m_Model->SetParameters( this->m_Model->GetParameters( ) - g );
}

// -------------------------------------------------------------------------
template< class _T >
PUJ_ML::Model::Base< _T >::
Base( )
{
  this->m_P = TCol::Zero( 1 );
}

// -------------------------------------------------------------------------
template< class _T >
typename PUJ_ML::Model::Base< _T >::
TCol& PUJ_ML::Model::Base< _T >::
GetParameters( )
{
  return( this->m_P );
}

// -------------------------------------------------------------------------
template< class _T >
const typename PUJ_ML::Model::Base< _T >::
TCol& PUJ_ML::Model::Base< _T >::
GetParameters( ) const
{
  return( this->m_P );
}

// -------------------------------------------------------------------------
template< class _T >
unsigned long PUJ_ML::Model::Base< _T >::
GetNumberOfParameters( ) const
{
  return( this->m_P.rows( ) );
}

// -------------------------------------------------------------------------
template< class _T >
void PUJ_ML::Model::Base< _T >::
SetParameters( const TCol& p )
{
  this->m_P = p;
}

// -------------------------------------------------------------------------
template< class _T >
typename PUJ_ML::Model::Base< _T >::
TMatrix PUJ_ML::Model::Base< _T >::
operator[]( const TMatrix& x )
{
  return( this->operator()( x ) );
}

// -------------------------------------------------------------------------
template< class _T >
void PUJ_ML::Model::Base< _T >::
_Out( std::ostream& o ) const
{
  o << this->m_P.rows( ) << " " << this->m_P.transpose( );
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
template class PUJ_ML_EXPORT PUJ_ML::Model::Base< float >;
template class PUJ_ML_EXPORT PUJ_ML::Model::Base< double >;
template class PUJ_ML_EXPORT PUJ_ML::Model::Base< long double >;

// eof - $RCSfile$
