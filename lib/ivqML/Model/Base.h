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
    template< class _TScalar >
    class Base
    {
    public:
      using Self     = Base;
      using TScalar  = _TScalar;
      using TNatural = unsigned long long;

      using TMat = Eigen::Matrix< TScalar, Eigen::Dynamic, Eigen::Dynamic >;
      using TCol = Eigen::Matrix< TScalar, Eigen::Dynamic, 1 >;
      using TRow = Eigen::Matrix< TScalar, 1, Eigen::Dynamic >;

      using TMatMap = Eigen::Map< TMat >;
      using TColMap = Eigen::Map< TCol >;
      using TRowMap = Eigen::Map< TRow >;

      using TMatCMap = Eigen::Map< const TMat >;
      using TColCMap = Eigen::Map< const TCol >;
      using TRowCMap = Eigen::Map< const TRow >;

    public:
      Base( const TNatural& n = 1 );
      virtual ~Base( ) = default;

      virtual bool has_backpropagation( ) const;

      /* TODO
         virtual void backpropagation(
         TScalar* G,
         TScalar* B,
         const TMatCMap& X, const TMatCMap& Y
         ) const;
      */

      virtual void random_fill( );

      _TScalar& operator[]( const TNatural& i );
      const _TScalar& operator[]( const TNatural& i ) const;

      virtual TNatural buffer_size( ) const;
      virtual TNatural number_of_parameters( ) const;
      virtual void set_number_of_parameters( const TNatural& p );

      virtual TNatural number_of_inputs( ) const = 0;
      virtual void set_number_of_inputs( const TNatural& p ) = 0;

      virtual TNatural number_of_outputs( ) const = 0;

      TMatMap matrix(
        const TNatural& r, const TNatural& c, const TNatural& o = 0
        );
      TMatCMap matrix(
        const TNatural& r, const TNatural& c, const TNatural& o = 0
        ) const;

      TColMap column( const TNatural& r, const TNatural& o = 0 );
      TColCMap column( const TNatural& r, const TNatural& o = 0 ) const;

      TRowMap row( const TNatural& c, const TNatural& o = 0 );
      TRowCMap row( const TNatural& c, const TNatural& o = 0 ) const;

    protected:
      virtual void _from_stream( std::istream& i );
      virtual void _to_stream( std::ostream& o ) const;

    protected:
      TCol m_P { 0 };

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
