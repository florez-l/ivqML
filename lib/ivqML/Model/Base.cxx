// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================

#include <ivqML/Model/Base.h>

#include <algorithm>
#include <cstring>

// -------------------------------------------------------------------------
template< class _TReal >
ivqML::Model::Base< _TReal >::
Base( const TNatural& n )
{
  this->set_size( n );
}

// -------------------------------------------------------------------------
template< class _TReal >
ivqML::Model::Base< _TReal >::
~Base( )
{
  if( this->m_P != nullptr )
    std::free( this->m_P );
}

// -------------------------------------------------------------------------
template< class _TReal >
const typename ivqML::Model::Base< _TReal >::
TNatural& ivqML::Model::Base< _TReal >::
size( ) const
{
  return( this->m_S );
}

// -------------------------------------------------------------------------
template< class _TReal >
void ivqML::Model::Base< _TReal >::
set_size( const TNatural& n )
{
  if( this->m_S != n )
  {
    this->m_S = n;
    this->init( );
  } // end if
}

// -------------------------------------------------------------------------
template< class _TReal >
void ivqML::Model::Base< _TReal >::
init( std::function< TReal( ) > g )
{
  if( this->m_P != nullptr )
    std::free( this->m_P );
  this->m_P = nullptr;
  if( this->m_S > 0 )
    this->m_P
      =
      reinterpret_cast< TReal* >( std::calloc( this->m_S, sizeof( TReal ) ) );
  if( this->m_P != nullptr )
    std::generate( this->m_P, this->m_P + this->m_S, g );
  else
    this->m_S = 0;
}

// -------------------------------------------------------------------------
template< class _TReal >
typename ivqML::Model::Base< _TReal >::
TReal ivqML::Model::Base< _TReal >::
_regularize(
  const TReal& J, TReal* G, const TReal& l1, const TReal& l2
  ) const
{
  TReal rJ = J;
  if( l1 != TReal( 0 ) || l2 != TReal( 0 ) )
  {
    TMatrixMap P( this->m_P, this->m_S, 1 );
    if( l1 != TReal( 0 ) )
      rJ += P.array( ).abs( ).sum( ) * l1;
    if( l2 != TReal( 0 ) )
      rJ += P.array( ).pow( 2 ).sum( ) * l2;

    TMatrixMap( G, this->m_S, 1 )
      =
      TMatrixMap( G, this->m_S, 1 )
      .binaryExpr(
        P,
        [&l1,&l2]( const TReal& g, const TReal& p ) -> TReal
        {
          TReal rg = g;
          if( l1 != TReal( 0 ) )
            rg
              +=
              ( p < TReal( 0 ) )
              ? -l1
              : ( ( p > TReal( 0 ) )? l1: TReal( 0 ) );
          if( l2 != TReal( 0 ) )
            rg += TReal( 2 ) * l2 * p;
          return( rg );
        }
        );
  } // end if
  return( rJ );
}

// -------------------------------------------------------------------------
template< class _TReal >
void ivqML::Model::Base< _TReal >::
_to_stream( std::ostream& o ) const
{
  o << this->m_S;
  for( TNatural i = 0; i < this->m_S; ++i )
    o << " " << *( this->m_P + i );
}

// -------------------------------------------------------------------------
namespace ivqML
{
  namespace Model
  {
    template class ivqML_EXPORT Base< float >;
    template class ivqML_EXPORT Base< double >;
    template class ivqML_EXPORT Base< long double >;
  } // end namespace
} // end namespace

// eof - $RCSfile$
