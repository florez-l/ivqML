// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================

#include <ivqML/Model/Regression/Linear.h>

// -------------------------------------------------------------------------
template< class _TReal, class _TNatural >
ivqML::Model::Regression::Linear< _TReal, _TNatural >::
Linear( const TNatural& n )
  : Superclass( n + 1 )
{
}

// -------------------------------------------------------------------------
template< class _TReal, class _TNatural >
ivqML::Model::Regression::Linear< _TReal, _TNatural >::
~Linear( )
{
}

// -------------------------------------------------------------------------
template< class _TReal, class _TNatural >
typename ivqML::Model::Regression::Linear< _TReal, _TNatural >::
TReal& ivqML::Model::Regression::Linear< _TReal, _TNatural >::
operator[]( const TNatural& i )
{
  static TReal _z;
  if( i < this->m_S )
    return( this->m_P[ i ] );
  else
  {
    _z = TReal( 0 );
    return( _z );
  } // end if
}

// -------------------------------------------------------------------------
template< class _TReal, class _TNatural >
const typename ivqML::Model::Regression::Linear< _TReal, _TNatural >::
TReal& ivqML::Model::Regression::Linear< _TReal, _TNatural >::
operator[]( const TNatural& i ) const
{
  static const TReal _z = TReal( 0 );
  if( i < this->m_S )
    return( this->m_P[ i ] );
  else
    return( _z );
}

// -------------------------------------------------------------------------
namespace ivqML
{
  namespace Model
  {
    namespace Regression
    {
      template class ivqML_EXPORT Linear< float, unsigned int >;
      template class ivqML_EXPORT Linear< float, unsigned long >;
      template class ivqML_EXPORT Linear< float, unsigned long long >;

      template class ivqML_EXPORT Linear< double, unsigned int >;
      template class ivqML_EXPORT Linear< double, unsigned long >;
      template class ivqML_EXPORT Linear< double, unsigned long long >;

      template class ivqML_EXPORT Linear< long double, unsigned int >;
      template class ivqML_EXPORT Linear< long double, unsigned long >;
      template class ivqML_EXPORT Linear< long double, unsigned long long >;
    } // end namespace
  } // end namespace
} // end namespace

// eof - $RCSfile$
