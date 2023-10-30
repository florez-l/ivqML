// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __ivqML__Cost__MSE__h__
#define __ivqML__Cost__MSE__h__

#include <ivqML/Cost/Base.h>
   
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
      MSE( const _M& m, const TX& iX, const TY& iY );
      virtual ~MSE( );

      virtual TResult operator()( ) const override;

    protected:
      TScalar* m_Dm { nullptr };
    };
  } // end namespace
} // end namespace

#include <ivqML/Cost/MSE.hxx>

#endif // __ivqML__Cost__MSE__h__

// eof - $RCSfile$
