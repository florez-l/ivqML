// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================

#include <ivqML/Model/Base.h>
#include <thread>

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
_R( const TReal& J, TReal* bG, const TReal& l1, const TReal& l2 ) const
{
  TReal rJ = TReal( 0 );
  if( l1 != TReal( 0 ) || l2 != 0 )
  {
    TNatural N = std::thread::hardware_concurrency( ) >> 2;
    N = ( N == 0 )? 1: N;
    TNatural tS = this->m_S / N;
    TNatural lS = this->m_S % N;
    N  = ( tS == 0 )? 0: N;
    N += ( lS != 0 )? 1: 0;

    std::vector< TReal > bJ( N, TReal( 0 ) );
    auto f = [ &l1, &l2, &bJ ]( TReal* P, TReal* G, const TNatural& S, const TNatural& i ) -> void
      {
        for( TNatural s = 0; s < S; ++s )
        {
          if( l1 != TReal( 0 ) )
          {
            bJ[ i ] += std::fabs( *( P + s ) ) * l1;
            *( G + s ) += TReal( ( *( P + s ) > TReal( 0 ) )? 1: ( ( *( P + s ) < TReal( 0 ) )? -1: 0 ) ) * l1;
          } // end if
          if( l2 != TReal( 0 ) )
          {
            bJ[ i ] += *( P + s ) * *( P + s ) * l2;
            *( G + s ) += TReal( 2 ) * *( P + s ) * l2;
          } // end if
        } // end if
      };

    if( N > 1 )
    {
      std::vector< std::thread > threads;
      TReal* P = this->m_P;
      TReal* G = bG;
      for( TNatural i = 0; i < N - 1; ++i )
      {
        threads.emplace_back( f, P, G, tS, i );
        P += tS;
        G += tS;
      } // end for
      threads.emplace_back( f, P, G, ( lS != 0 )? lS: tS, N - 1 );

      for( std::thread& t: threads )
        t.join( );
    }
    else
      f( this->m_P, bG, this->m_S, 0 );
    
    rJ = std::accumulate( bJ.begin( ), bJ.end( ), 0 );
  } // end if
  return( J + rJ );
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
