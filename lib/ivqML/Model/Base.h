// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __ivqML__Model__Base__h__
#define __ivqML__Model__Base__h__

#include <ivqML/Config.h>

namespace ivqML
{
  namespace Model
  {
    /**
     */
    template< class _TReal, class _TNatural = unsigned long long >
    class Base
    {
    public:
      using TReal    = _TReal;
      using TNatural = _TNatural;
      using Self     = Base;
      using TMatrix
      = Eigen::Matrix< TReal, Eigen::Dynamic, Eigen::Dynamic >;
      using TColumn   = Eigen::Matrix< TReal, Eigen::Dynamic, 1 >;
      using TRow      = Eigen::Matrix< TReal, 1, Eigen::Dynamic >;
      using TMap      = Eigen::Map< TMatrix >;
      using TConstMap = Eigen::Map< const TMatrix >;

    public:
      Base( const TNatural& n = 1 );
      virtual ~Base( );

      TReal& operator[]( const TNatural& i );
      const TReal& operator[]( const TNatural& i ) const;

      template< class _Tw >
      Self& operator+=( const Eigen::EigenBase< _Tw >& w );

      template< class _Tw >
      Self& operator-=( const Eigen::EigenBase< _Tw >& w );

      const TNatural& size( ) const;
      void init( );

    protected:
      virtual void _resize( const TNatural& n );

      template< class _TG >
      void _regularize(
        Eigen::EigenBase< _TG >& G, const TReal& L1, const TReal& L2
        ) const;

      virtual void _to_stream( std::ostream& o ) const;

    protected:
      TReal*   m_P { nullptr };
      TNatural m_S { 0 };

    public:
      friend std::ostream& operator<<( std::ostream& o, const Self& m )
        {
          m._to_stream( o );
          return( o );
        }
    };
  } // end namespace
} // end namespace

#include <ivqML/Model/Base.hxx>

#endif // __ivqML__Model__Base__h__

// eof - Base.h
