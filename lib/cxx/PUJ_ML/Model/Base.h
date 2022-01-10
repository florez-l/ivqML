// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __PUJ_ML__Model__Base__h__
#define __PUJ_ML__Model__Base__h__

#include <cassert>
#include <Eigen/Core>
#include <PUJ_ML/Export.h>

namespace PUJ_ML
{
  namespace Model
  {
    /**
     */
    template< class _T >
    class Base
    {
    public:
      using Self    = Base;
      using TScalar = _T;
      using TMatrix = Eigen::Matrix< _T, Eigen::Dynamic, Eigen::Dynamic >;
      using TRow    = Eigen::Matrix< _T, 1, Eigen::Dynamic >;
      using TCol    = Eigen::Matrix< _T, Eigen::Dynamic, 1 >;

    public:
      /**
       */
      class Cost
      {
      public:
        Cost( Self* model, const TMatrix& X, const TMatrix& Y );
        virtual ~Cost( ) = default;
        
        const TCol& GetParameters( ) const;

        virtual _T Compute( TCol* g = nullptr ) const = 0;
        virtual void Update( const TCol& g ) const;

      protected:
        Self* m_Model { nullptr };

        const TMatrix* m_X { nullptr };
        const TMatrix* m_Y { nullptr };
      };

    public:
      Base( );
      virtual ~Base( ) = default;

      TCol& GetParameters( );
      const TCol& GetParameters( ) const;
      unsigned long GetNumberOfParameters( ) const;

      template< class _I >
      void SetParameters( _I b, _I e );
      void SetParameters( const TCol& p );

      virtual TMatrix operator()( const TMatrix& x ) = 0;
      virtual TMatrix operator[]( const TMatrix& x );

    protected:
      void _Out( std::ostream& o ) const;
      void _In( std::istream& i );

    protected:
      TCol m_P;

    public:
      friend std::ostream& operator<<( std::ostream& o, const Self& m )
        {
          m._Out( o );
          return( o );
        }
      friend std::istream& operator>>( std::istream& i, Self& m )
        {
          m._In( i );
          return( i );
        }
    };
  } // end namespace
} // end namespace

// -------------------------------------------------------------------------
template< class _T >
template< class _I >
void PUJ_ML::Model::Base< _T >::
SetParameters( _I b, _I e )
{
  this->m_P.resize( std::distance( b, e ) );
  unsigned long long k = 0;
  for( auto i = b; i != e; ++i )
    this->m_P( k++ ) = _T( *i );
}

#endif // __PUJ_ML__Model__Base__h__

// eof - $RCSfile$
