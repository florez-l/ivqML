// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __ivqML__Model__Base__hxx__
#define __ivqML__Model__Base__hxx__

// -------------------------------------------------------------------------
template< class _TReal, class _TNatural >
template< class _Tw >
typename ivqML::Model::Base< _TReal, _TNatural >::
Self& ivqML::Model::Base< _TReal, _TNatural >::
operator+=( const Eigen::EigenBase< _Tw >& w )
{
  if( w.size( ) == this->m_S )
    TMatMap( this->m_P, w.rows( ), w.cols( ) )
      +=
      w.derived( ).template cast< TReal >( );
  return( *this );
}

// -------------------------------------------------------------------------
template< class _TReal, class _TNatural >
template< class _Tw >
typename ivqML::Model::Base< _TReal, _TNatural >::
Self& ivqML::Model::Base< _TReal, _TNatural >::
operator-=( const Eigen::EigenBase< _Tw >& w )
{
  if( w.size( ) == this->m_S )
    TMatMap( this->m_P, w.rows( ), w.cols( ) )
      -=
      w.derived( ).template cast< TReal >( );
  return( *this );
}

// -------------------------------------------------------------------------
template< class _TReal, class _TNatural >
template< class _TG >
void ivqML::Model::Base< _TReal, _TNatural >::
_regularize(
  Eigen::EigenBase< _TG >& G, const TReal& L1, const TReal& L2
  ) const
{
  static const TReal _0 = TReal( 0 );
  static const TReal _2 = TReal( 2 );

  TNatural k = 0;
  for( TNatural r = 0; r < G.rows( ); ++r )
  {
    for( TNatural c = 0; c < G.cols( ); ++c )
    {
      G.derived( )( r, c ) +=
        ( typename _TG::Scalar )(
          ( ( this->m_P[ k ] > _0 )? L1: ( ( this->m_P[ k ] < _0 )? -L1: _0 ) )
          +
          ( _2 * L2 * this->m_P[ k ] )
          );
      k++;
    } // end for
  } // end for
}

#endif // __ivqML__Model__Base__hxx__

// eof - Base.hxx
