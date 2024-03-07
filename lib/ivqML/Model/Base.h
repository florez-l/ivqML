// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __ivqML__Model__Base__h__
#define __ivqML__Model__Base__h__

#include <ivqML/Config.h>
#include <istream>
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

      virtual TNatural number_of_parameters( ) const;
      virtual void set_number_of_parameters( const TNatural& p );

      virtual TNatural number_of_inputs( ) const = 0;
      virtual void set_number_of_inputs( const TNatural& p ) = 0;

      virtual TNatural number_of_outputs( ) const = 0;

    protected:
      virtual void _map(
        TMap& map,
        const TNatural& r, const TNatural& c, const TNatural& o = 0
        );
      virtual void _from_stream( std::istream& i );
      virtual void _to_stream( std::ostream& o ) const;

    protected:
      TScalar* m_Parameters { nullptr };
      TNatural m_Size       { 0 };

    public:
      friend std::istream& operator>>( std::istream& i, Self& m )
        {
          m._from_stream( i );
          return( i );
        }
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
