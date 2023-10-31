// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __ivqML__Optimizer__Base__h__
#define __ivqML__Optimizer__Base__h__

#include <functional>
#include <map>

namespace ivqML
{
  namespace Optimizer
  {
    /**
     */
    template< class _C >
    class Base
    {
    public:
      using Self = Base;
      using TCost = _C;
      using TModel = typename _C::TModel;
      using TDX = typename _C::TDX;
      using TDY = typename _C::TDY;
      using TX = typename _C::TX;
      using TY = typename _C::TY;
      using TScalar = typename _C::TScalar;
      using TNatural = typename _C::TNatural;
      using TMatrix = typename _C::TMatrix;
      using TMap = typename _C::TMap;
      using TConstMap = typename _C::TConstMap;
      using TResult = typename _C::TResult;

      using TDebug =
        std::function< bool( const TScalar&, const TScalar&, const TModel*, const TNatural& ) >;

    public:
      Base( TModel& m, const TX& iX, const TY& iY );
      virtual ~Base( ) = default;

      template< class _V >
      bool parameter( _V& v, const std::string& n ) const;

      template< class _V >
      void configure_parameter( const std::string& n, const _V& v );

      template< class _V >
      void set_parameter( const std::string& n, const _V& v );

      void set_debug( TDebug d );

      virtual void fit( ) = 0;

    protected:
      TModel*   m_M { nullptr };
      const TX* m_X { nullptr };
      const TY* m_Y { nullptr };

      std::map< std::string, std::string > m_P;

      TDebug m_D
        {
          [](
            const TScalar&, const TScalar&, const TModel*, const TNatural&
            ) -> bool
          {
            return( false );
          }
        };
    };
  } // end namespace
} // end namespace

#include <ivqML/Optimizer/Base.hxx>

#endif // __ivqML__Optimizer__Base__h__

// eof - $RCSfile$
