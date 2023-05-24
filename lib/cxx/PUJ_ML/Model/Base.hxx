// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __PUJ_ML__Model__Base__hxx__
#define __PUJ_ML__Model__Base__hxx__

// -------------------------------------------------------------------------
template< class _D, class _R >
PUJ_ML::Model::Base< _D, _R >::
Base( const unsigned long long& n )
{
  this->init( n );
}

// -------------------------------------------------------------------------
template< class _D, class _R >
unsigned long long PUJ_ML::Model::Base< _D, _R >::
number_of_parameters( ) const
{
  return( this->m_P.size( ) );
}

// -------------------------------------------------------------------------
template< class _D, class _R >
unsigned long long PUJ_ML::Model::Base< _D, _R >::
number_of_inputs( ) const
{
  return( this->number_of_parameters( ) );
}

// -------------------------------------------------------------------------
template< class _D, class _R >
void PUJ_ML::Model::Base< _D, _R >::
init( const unsigned long long& n )
{
  this->m_P.resize( n, TReal( 0 ) );
  this->m_P.shrink_to_fit( );
}

// -------------------------------------------------------------------------
template< class _D, class _R >
typename PUJ_ML::Model::Base< _D, _R >::
TReal& PUJ_ML::Model::Base< _D, _R >::
operator()( const unsigned long long& i )
{
  static TReal zero;
  if( i < this->m_P.size( ) )
    return( this->m_P[ i ] );
  else
  {
    zero = 0;
    return( zero );
  } // end if
}

// -------------------------------------------------------------------------
template< class _D, class _R >
const typename PUJ_ML::Model::Base< _D, _R >::
TReal& PUJ_ML::Model::Base< _D, _R >::
operator()( const unsigned long long& i ) const
{
  static const TReal zero = 0;
  if( i < this->m_P.size( ) )
    return( this->m_P[ i ] );
  else
    return( zero );
}

// -------------------------------------------------------------------------
template< class _D, class _R >
template< class _Y, class _X >
void PUJ_ML::Model::Base< _D, _R >::
threshold( Eigen::EigenBase< _Y >& Y, const Eigen::EigenBase< _X >& X ) const
{
  static_cast< const _D* >( this )->evaluate( Y, X );
}

// -------------------------------------------------------------------------
template< class _D, class _R >
void PUJ_ML::Model::Base< _D, _R >::
move_parameters( const std::vector< TReal >& dir, const TReal& coeff )
{
  for( unsigned long long i = 0; i < this->m_P.size( ); ++i )
    this->m_P[ i ] += dir[ i ] * coeff;
}

// -------------------------------------------------------------------------
template< class _D, class _R >
void PUJ_ML::Model::Base< _D, _R >::
_from_stream( std::istream& i )
{
}

// -------------------------------------------------------------------------
template< class _D, class _R >
void PUJ_ML::Model::Base< _D, _R >::
_to_stream( std::ostream& o ) const
{
  o << this->m_P.size( );
  for( const TReal& v: this->m_P )
    o << " " << v;
}

#endif // __PUJ_ML__Model__Base__hxx__

// eof - $RCSfile$
