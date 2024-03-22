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
    template< class _TScl >
    class Base
    {
    public:
      using Self = Base;
      using TScl = _TScl;
      using TNat = unsigned long long;

      using TMat = Eigen::Matrix< TScl, Eigen::Dynamic, Eigen::Dynamic >;
      using TCol = Eigen::Matrix< TScl, Eigen::Dynamic, 1 >;
      using TRow = Eigen::Matrix< TScl, 1, Eigen::Dynamic >;

      using TMatMap = Eigen::Map< TMat >;
      using TColMap = Eigen::Map< TCol >;
      using TRowMap = Eigen::Map< TRow >;

      using TMatCMap = Eigen::Map< const TMat >;
      using TColCMap = Eigen::Map< const TCol >;
      using TRowCMap = Eigen::Map< const TRow >;

    public:
      Base( const TNat& n = 1 );
      virtual ~Base( );

      virtual void random_fill( );

      _TScl& operator[]( const TNat& i );
      const _TScl& operator[]( const TNat& i ) const;

      virtual TNat buffer_size( ) const;
      virtual TNat number_of_parameters( ) const;
      virtual void set_number_of_parameters( const TNat& p );

      virtual TNat number_of_inputs( ) const = 0;
      virtual void set_number_of_inputs( const TNat& p ) = 0;

      virtual TNat number_of_outputs( ) const = 0;

      TMatMap matrix( const TNat& r, const TNat& c, const TNat& o = 0 );
      TMatCMap matrix( const TNat& r, const TNat& c, const TNat& o = 0 ) const;

      TColMap column( const TNat& r, const TNat& o = 0 );
      TColCMap column( const TNat& r, const TNat& o = 0 ) const;

      TRowMap row( const TNat& c, const TNat& o = 0 );
      TRowCMap row( const TNat& c, const TNat& o = 0 ) const;

      virtual bool has_backpropagation( ) const;

      template< class _TInputX, class _TInputY >
      void backpropagation(
        TScl* G,
        TScl* B,
        const Eigen::EigenBase< _TInputX >& iX,
        const Eigen::EigenBase< _TInputY >& iY
        ) const
        {
          // Do nothing!
        }

    protected:
      virtual void _from_stream( std::istream& i );
      virtual void _to_stream( std::ostream& o ) const;

    private:
      Base( const Self& ) = delete;
      Self& operator=( const Self& ) = delete;

    private:
      TScl* m_P { nullptr };
      TNat  m_S { 0 };

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

#endif // __ivqML__Model__Base__h__

// eof - $RCSfile$
