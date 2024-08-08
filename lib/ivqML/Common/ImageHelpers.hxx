// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __ivqML__Common__ImageHelpers__hxx__
#define __ivqML__Common__ImageHelpers__hxx__

// -------------------------------------------------------------------------
template< class _I, class _M >
_M ivqML::Common::ImageHelpers::meshgrid( const _I* image )
{
  using _S = typename _M::Scalar;

  // TODO: this is **almost** dimension-generic -> work more on linSpaced!
  auto roi = image->GetRequestedRegion( );
  auto spac = image->GetSpacing( );
  auto orig = image->GetOrigin( );
  unsigned long long w = roi.GetSize( )[ 0 ];
  unsigned long long h = roi.GetSize( )[ 1 ];
  unsigned long long s = w * h;

  _M S = Eigen::Map< const Eigen::Matrix< typename decltype( spac )::ValueType, Eigen::Dynamic, Eigen::Dynamic > >( spac.data( ), 1, _I::ImageDimension ).template cast< _S >( );
  _M O = Eigen::Map< const Eigen::Matrix< typename decltype( orig )::ValueType, Eigen::Dynamic, Eigen::Dynamic > >( orig.data( ), 1, _I::ImageDimension ).template cast< _S >( );

  _M rows( h, 1 ), cols( w, 1 );
  rows.col( 0 ).setLinSpaced( h, 0, h - 1 );
  cols.col( 0 ).setLinSpaced( w, 0, w - 1 );
  _M G( s, _I::ImageDimension );
  G
    <<
    cols.replicate( 1, h ).reshaped( s, 1 ),
    rows.replicate( 1, w ).transpose( ).reshaped( s, 1 );
  G.array( ).rowwise( ) *= S.array( ).row( 0 );
  G.array( ).rowwise( ) += O.array( ).row( 0 );
  return( G );
}

#endif // __ivqML__Common__ImageHelpers__hxx__

// eof - $RCSfile$
