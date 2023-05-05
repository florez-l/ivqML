// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __PUJ_ML__Model__Base__hxx__
#define __PUJ_ML__Model__Base__hxx__

// -------------------------------------------------------------------------
template< class _R >
PUJ_ML::Model::Base< _R >::
Base( const unsigned long long& n )
{
  this->set_number_of_parameters( n );
}

// -------------------------------------------------------------------------
template< class _R >
unsigned long long PUJ_ML::Model::Base< _R >::
number_of_parameters( ) const
{
  return( this->m_P.size( ) );
}

// -------------------------------------------------------------------------
template< class _R >
void PUJ_ML::Model::Base< _R >::
set_number_of_parameters( const unsigned long long& n )
{
  this->m_P.resize( n, _R( 0 ) );
  this->m_P.shrink_to_fit( );
}

// -------------------------------------------------------------------------
template< class _R >
_R& PUJ_ML::Model::Base< _R >::
operator()( const unsigned long long& i )
{
  static _R zero;
  if( i < this->m_P.size( ) )
    return( this->m_P[ i ] );
  else
  {
    zero = 0;
    return( zero );
  } // end if
}

// -------------------------------------------------------------------------
template< class _R >
const _R& PUJ_ML::Model::Base< _R >::
operator()( const unsigned long long& i ) const
{
  static const _R zero = 0;
  if( i < this->m_P.size( ) )
    return( this->m_P[ i ] );
  else
    return( zero );
}

// -------------------------------------------------------------------------
template< class _R >
_R& PUJ_ML::Model::Base< _R >::
operator()( const unsigned long long& i, const unsigned long long& j )
{
  return( this->operator()( i ) );
}

// -------------------------------------------------------------------------
template< class _R >
const _R& PUJ_ML::Model::Base< _R >::
operator()( const unsigned long long& i, const unsigned long long& j ) const
{
  return( this->operator()( i ) );
}

// -------------------------------------------------------------------------
template< class _R >
template< class _Y, class _X >
void PUJ_ML::Model::Base< _R >::
evaluate( Eigen::EigenBase< _Y >& Y, const Eigen::EigenBase< _X >& X ) const
{
}

// -------------------------------------------------------------------------
template< class _R >
template< class _Y, class _X >
void PUJ_ML::Model::Base< _R >::
threshold( Eigen::EigenBase< _Y >& Y, const Eigen::EigenBase< _X >& X ) const
{
  this->evaluate( Y, X );
}

// -------------------------------------------------------------------------
template< class _R >
void PUJ_ML::Model::Base< _R >::
_to_stream( std::ostream& o ) const
{
  o << this->m_P.size( );
  for( const _R& v: this->m_P )
    o << " " << v;
}

#endif // __PUJ_ML__Model__Base__hxx__

// eof - $RCSfile$
