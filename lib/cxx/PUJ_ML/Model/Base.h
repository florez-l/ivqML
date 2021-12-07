// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __PUJ_ML__Model__Base__h__
#define __PUJ_ML__Model__Base__h__

#include <Eigen/Core>

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
      using TColumn = Eigen::Matrix< _T, Eigen::Dynamic, 1 >;

    public:
      /**
       */
      class Cost
      {
      public:
        enum ERegularization
        {
          RidgeReg = 0,
          LASSOReg
        };

      public:
        Cost( Self* model, const TMatrix& X, const TMatrix& Y );
        virtual ~Cost( ) = default;

        const ERegularization& regularization( ) const;
        void set_ridge_regularization( );
        void set_LASSO_regularization( );

        const _T& lambda( ) const;
        void set_lambda( const _T& l );

        virtual _T operator()( _T* g = nullptr ) const;

      protected:
        Self* m_Model;

        const TMatrix* m_X;
        const TMatrix* m_Y;

        ERegularization m_Regularization;
        _T m_Lambda;
      };

    public:
      Base( );
      virtual ~Base( ) = default;

      TRow& parameters( );
      const TRow& parameters( ) const;
      unsigned long number_of_parameters( ) const;

      template< class _I >
      void set_parameters( _I b, _I e );
      void set_parameters( const TRow& p );

      virtual TColumn operator()( const TMatrix& x ) = 0;
      virtual TColumn operator[]( const TMatrix& x );

    protected:
      void _Out( std::ostream& o ) const;
      void _In( std::istream& i );

    protected:
      TRow m_P;

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
set_parameters( _I b, _I e )
{
  this->m_P.resize( std::distance( b, e ) );
  unsigned long long k = 0;
  for( auto i = b; i != e; ++i )
    this->m_P( k++ ) = *i;
}

#endif // __PUJ_ML__Model__Base__h__

// eof - $RCSfile$
