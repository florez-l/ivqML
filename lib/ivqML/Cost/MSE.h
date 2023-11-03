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

      ivqML_Cost_Typedefs;

    public:
      MSE( const _M& m, const TX& iX, const TY& iY );
      virtual ~MSE( ) override = default;

      virtual TResult operator()( ) override;

    protected:
      TMatrix m_D;
    };
  } // end namespace
} // end namespace

#include <ivqML/Cost/MSE.hxx>

#endif // __ivqML__Cost__MSE__h__

// eof - $RCSfile$
