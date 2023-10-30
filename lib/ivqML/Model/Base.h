// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __ivqML__Model__Base__h__
#define __ivqML__Model__Base__h__

#include <ivqML/Config.h>
#include <ostream>

namespace ivqML
{
  namespace Model
  {
    /**
     */
    template< class _S >
    class Base
    {
    public:
      using Self = Base;
      using TScalar = _S;
      using TNatural = unsigned long long;

      using TMatrix = Eigen::Matrix< _S, Eigen::Dynamic, Eigen::Dynamic >;
      using TMap = Eigen::Map< TMatrix >;
      using TConstMap = Eigen::Map< const TMatrix >;

    public:
      Base( const TNatural& n = 1 );
      virtual ~Base( );

      template< class _O >
      Base( const _O& other );

      template< class _O >
      Self& operator=( const _O& other );

      virtual void random_fill( );

      _S& operator[]( const TNatural& i );
      const _S& operator[]( const TNatural& i ) const;

      template< class _D >
      Self& operator+=( const Eigen::EigenBase< _D >& d );

      template< class _D >
      Self& operator-=( const Eigen::EigenBase< _D >& d );

      const TNatural& number_of_parameters( ) const;
      void set_number_of_parameters( const TNatural& p );

      _S* begin( );
      const _S* begin( ) const;

      _S* end( );
      const _S* end( ) const;

    protected:
      virtual void _to_stream( std::ostream& o ) const;

    protected:
      _S* m_T { nullptr };
      TNatural m_P { 0 };

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

// eof - $RCSfile$
