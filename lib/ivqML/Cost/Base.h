// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __ivqML__Cost__Base__h__
#define __ivqML__Cost__Base__h__

/* TODO
   #include <Eigen/Core>
   #include <cmath>
   #include <ostream>
   #include <limits>
*/
   
namespace ivqML
{
  namespace Cost
  {
    /**
     */
    template< class _M, class _X, class _Y >
    class Base
    {
    public:
      using Self = Base;
      using TModel = _M;
      using TDX = _X;
      using TDY = _Y;
      using TX = Eigen::EigenBase< _X >;
      using TY = Eigen::EigenBase< _Y >;
      using TScalar = typename _M::TScalar;
      using TNatural = typename _M::TNatural;
      using TMatrix = typename _M::TMatrix;
      using TMap = typename _M::TMap;
      using TConstMap = typename _M::TConstMap;

      using TResult = std::pair< TScalar, const TScalar* >;

    public:
      Base( const _M& m, const _X& iX, const _Y& iY )
        : m_M( &m ),
          m_X( &iX ),
          m_Y( &iY )
        {
          this->m_G = new TScalar[ m.number_of_parameters( ) ];
          this->m_Ym = new TScalar[ iY.derived( ).size( ) ];
        }
      virtual ~Base( )
        {
          if( this->m_G != nullptr )
            delete this->m_G;
          if( this->m_Ym != nullptr )
            delete this->m_Ym;
        }
      
      virtual TResult operator()( ) const = 0;


    protected:
      const _M* m_M { nullptr };
      const TX* m_X { nullptr };
      const TY* m_Y { nullptr };

      TScalar* m_G  { nullptr };
      TScalar* m_Ym { nullptr };
    };
  } // end namespace
} // end namespace

namespace ivqML
{
  namespace Cost
  {
    /**
     */
    template< class _M, class _X = typename _M::TMatrix, class _Y = typename _M::TMatrix >
    class MSE
      : public ivqML::Cost::Base< _M, _X, _Y >
    {
    public:
      using Self = MSE;
      using Superclass = ivqML::Cost::Base< _M, _X, _Y >;
      using TModel = typename Superclass::TModel;
      using TDX = typename Superclass::TDX;
      using TDY = typename Superclass::TDY;
      using TX = typename Superclass::TX;
      using TY = typename Superclass::TY;
      using TScalar = typename Superclass::TScalar;
      using TNatural = typename Superclass::TNatural;
      using TMatrix = typename Superclass::TMatrix;
      using TMap = typename Superclass::TMap;
      using TConstMap = typename Superclass::TConstMap;
      using TResult = typename Superclass::TResult;

    public:
      MSE( const _M& m, const _X& iX, const _Y& iY )
        : Superclass( m, iX, iY )
        {
          this->m_Dm = new TScalar[ iX.rows( ) * ( iX.cols( ) + 1 ) ];
        }
      virtual ~MSE( )
        {
          if( this->m_Dm != nullptr )
            delete this->m_Dm;
        }

      virtual TResult operator()( ) const override
        {
          auto X = this->m_X->derived( ).template cast< TScalar >( );
          auto Y = this->m_Y->derived( ).template cast< TScalar >( );
          auto Ym = TMap( this->m_Ym, Y.rows( ), Y.cols( ) );
          auto Dm = TMap( this->m_Dm, X.rows( ), X.cols( ) + 1 );

          this->m_M->operator()( Ym, X );
          this->m_M->operator()( Dm, X, true );

          auto D = Ym - Y;
          TMap( this->m_G, 1, this->m_M->number_of_parameters( ) ) =
            ( Dm.array( ).colwise( ) * D.col( 0 ).array( ) )
            .colwise( ).mean( ) * TScalar( 2 );
          return( std::make_pair( D.array( ).pow( 2 ).mean( ), this->m_G ) );
        }
    protected:
      TScalar* m_Dm { nullptr };
    };
  } // end namespace
} // end namespace



#endif // __ivqML__Cost__Base__h__

// eof - $RCSfile$
