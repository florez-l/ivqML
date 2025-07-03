// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================

#include <ivqML/Common/MeanShift.h>

#include <cstring>

// -------------------------------------------------------------------------
template< class _TReal, class _TNatural >
bool ivqML::Common::MeanShift< _TReal, _TNatural >::SShiftCmp::
operator()( const TColumnMap& a, const TColumnMap& b ) const
{
  return(
    std::lexicographical_compare(
      a.begin( ), a.end( ), b.begin( ), b.end( )
      )
    );
}

// -------------------------------------------------------------------------
template< class _TReal, class _TNatural >
ivqML::Common::MeanShift< _TReal, _TNatural >::
MeanShift( TReal* data, const TNatural& dims, const TNatural& samples )
{
  this->_init( );
  this->_allocate( dims * samples );
  this->_go( data, this->m_ShiftedData, dims, samples );
}

// -------------------------------------------------------------------------
template< class _TReal, class _TNatural >
ivqML::Common::MeanShift< _TReal, _TNatural >::
~MeanShift( )
{
  this->_free( );
}

// -------------------------------------------------------------------------
template< class _TReal, class _TNatural >
void ivqML::Common::MeanShift< _TReal, _TNatural >::
_init( )
{
  this->m_DistanceError
    =
    std::pow(
      10, std::log10( std::numeric_limits< TReal >::epsilon( ) ) * 0.5
      );
  this->m_Kernel
    =
    []( const TColumnMap& a, const TColumnMap& b ) -> TReal
    {
      TReal e = ( a - b ).array( ).pow( 2 ).sum( ) / TReal( -4.5 );
      return( std::exp( e ) );
    };
}

// -------------------------------------------------------------------------
template< class _TReal, class _TNatural >
void ivqML::Common::MeanShift< _TReal, _TNatural >::
_allocate( const TNatural& S )
{
  this->_free( );
  this->m_ShiftedData
    =
    reinterpret_cast< TReal* >( std::calloc( S, sizeof( TReal ) ) );
}

// -------------------------------------------------------------------------
template< class _TReal, class _TNatural >
void ivqML::Common::MeanShift< _TReal, _TNatural >::
_free( )
{
  if( this->m_ShiftedData != nullptr )
    std::free( this->m_ShiftedData );
  this->m_ShiftedData = nullptr;
}

// -------------------------------------------------------------------------
template< class _TReal, class _TNatural >
void ivqML::Common::MeanShift< _TReal, _TNatural >::
_go( TReal* I, TReal* O, const TNatural& N, const TNatural& M )
{
  this->m_MeansMap.clear( );
  this->m_ShiftedMeansMap.clear( );

  for( TNatural i = 0; i < M; ++i )
  {
    std::cout << i <<  "/" << M << std::endl;
    TColumnMap xi( I + ( i * N ), N, 1 );
    TColumnMap xo( O + ( i * N ), N, 1 );

    TReal* mb = nullptr;
    if( this->m_MeansMap.size( ) > 0 )
    {
      auto mIt = this->m_MeansMap.lower_bound( xi );
      if( mIt == this->m_MeansMap.end( ) )
        mb = this->m_MeansMap.rbegin( )->second.data( );
      else
        mb = mIt->second.data( );
      if(
        this->m_DistanceError
        <
        std::sqrt( ( xi - TColumnMap( mb, N, 1 ) ).array( ).pow( 2 ).sum( ) )
        )
        mb = nullptr;
      xo = ( mb != nullptr )? TColumnMap( mb, N, 1 ): xi;
    }
    else
      xo = xi;

    if( mb == nullptr )
    {
      bool stop = false;
      TColumn xs( N, 1 );
      TNatural k = 0;
      while( !stop )
      {
        xs.fill( 0 );
        TReal W = 0;
        for( TNatural j = 0; j < M; ++j )
        {
          TColumnMap xj( I + ( j * N ), N, 1 );
          TReal w = this->m_Kernel( xo, xj );
          xs += xj * w;
          W += w;
        } // end for
        if( W != TReal( 0 ) )
          xs /= W;
        else
          xs.fill( 0 );
        TReal d = std::sqrt( ( xo - xs ).array( ).pow( 2 ).sum( ) );
        stop
          =
          !( this->m_DistanceError < d )
          ||
          !( ++k < this->m_MaximumNumberOfIterations );
        xo = xs;
      } // end while

      bool new_mean = ( this->m_ShiftedMeansMap.size( ) == 0 );
      if( !new_mean )
      {
        TReal* mb = nullptr;
        auto smIt = this->m_ShiftedMeansMap.lower_bound( xo );
        if( smIt == this->m_ShiftedMeansMap.end( ) )
          mb
            =
            const_cast< TReal* >(
              this->m_ShiftedMeansMap.rbegin( )->first.data( )
              );
        else
          mb = const_cast< TReal* >( smIt->first.data( ) );
        new_mean
          =
          (
            this->m_DistanceError
            <
            std::sqrt(
              ( xo - TColumnMap( mb, N, 1 ) ).array( ).pow( 2 ).sum( )
              )
            );

        if( !new_mean )
        {
          if( smIt == this->m_ShiftedMeansMap.end( ) )
            this->m_ShiftedMeansMap.rbegin( )->second.push_back( xi );
          else
            smIt->second.push_back( xi );
        } // end if
      } // end if

      if( new_mean )
      {
        this->m_MeansMap.insert( std::make_pair( xi, xo ) );
        this->m_ShiftedMeansMap.insert(
          std::make_pair( xo, std::vector< TColumnMap >( ) )
          ).first->second.push_back( xi );
      } // end if
    } // end if
  } // end for
}

// -------------------------------------------------------------------------
namespace ivqML
{
  namespace Common
  {
    template class ivqML_EXPORT MeanShift< float, unsigned short >;
    template class ivqML_EXPORT MeanShift< double, unsigned short >;
    template class ivqML_EXPORT MeanShift< long double, unsigned short >;

    template class ivqML_EXPORT MeanShift< float, unsigned int >;
    template class ivqML_EXPORT MeanShift< double, unsigned int >;
    template class ivqML_EXPORT MeanShift< long double, unsigned int >;

    template class ivqML_EXPORT MeanShift< float, unsigned long >;
    template class ivqML_EXPORT MeanShift< double, unsigned long >;
    template class ivqML_EXPORT MeanShift< long double, unsigned long >;

    template class ivqML_EXPORT MeanShift< float, unsigned long long >;
    template class ivqML_EXPORT MeanShift< double, unsigned long long >;
    template class ivqML_EXPORT MeanShift< long double, unsigned long long >;
  } // end namespace
} // end namespace

// eof - $RCSfile$
