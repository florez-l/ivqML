// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================

#include <ivqML/Helpers/SplitDatasetWithBinaryLabeling.h>
#include <random>

// -------------------------------------------------------------------------
template< class _TR >
void ivqML::Helpers::SplitDatasetWithBinaryLabeling< _TR >::
init( const _TR& v, const _TIdx& r, const _TIdx& c )
{
  this->Z.clear( );
  this->O.clear( );
  this->operator()( v, r, c );
}

// -------------------------------------------------------------------------
template< class _TR >
void ivqML::Helpers::SplitDatasetWithBinaryLabeling< _TR >::
 operator()( const _TR& v, const _TIdx& r, const _TIdx& c )
{
  if     ( v == 0 ) this->Z.push_back( r );
  else if( v == 1 ) this->O.push_back( r );
}

// -------------------------------------------------------------------------
template< class _TR >
void ivqML::Helpers::SplitDatasetWithBinaryLabeling< _TR >::
finish( const _TR& s )
{
  // Shuffle both labels
  std::random_device rand_dev;
  std::mt19937 rang_gen( rand_dev( ) );
  std::shuffle( this->Z.begin( ), this->Z.end( ), rang_gen );
  std::shuffle( this->O.begin( ), this->O.end( ), rang_gen );

  // Compute sizes
  unsigned long long n = std::min( this->Z.size( ), this->O.size( ) );
  unsigned long long n_tr = ( unsigned long long )( _TR( n ) * s );

  // Prepare final indices
  this->Tr.clear( );
  this->Te.clear( );

  this->Tr.insert( this->Tr.end( ), this->Z.begin( ), this->Z.begin( ) + n_tr );
  this->Tr.insert( this->Tr.end( ), this->O.begin( ), this->O.begin( ) + n_tr );
  std::shuffle( this->Tr.begin( ), this->Tr.end( ), rang_gen );

  if( n_tr < n )
  {
    this->Te.insert( this->Te.end( ), this->Z.begin( ) + n_tr, this->Z.begin( ) + n );
    this->Te.insert( this->Te.end( ), this->O.begin( ) + n_tr, this->O.begin( ) + n );
    std::shuffle( this->Te.begin( ), this->Te.end( ), rang_gen );
  } // end if
}

// -------------------------------------------------------------------------
namespace ivqML
{
  namespace Helpers
  {
    template struct ivqML_EXPORT SplitDatasetWithBinaryLabeling< float >;
    template struct ivqML_EXPORT SplitDatasetWithBinaryLabeling< double >;
    template struct ivqML_EXPORT SplitDatasetWithBinaryLabeling< long double >;
  } // end namespace
} // end namespace

// eof - $RCSfile$
