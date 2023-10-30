// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __ivqML__Optimizer__Base__h__
#define __ivqML__Optimizer__Base__h__

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

    public:
      Base( TModel& m, const TX& iX, const TY& iY )
        : m_M( &m ),
          m_X( &iX ),
          m_Y( &iY )
        {
          this->m_P[ "lambda" ] = "0";
          this->m_P[ "regularization" ] = "ridge";
          this->m_P[ "max_iterations" ] = "100000";
          this->m_P[ "debug_iterations" ] = "100";
        }

      virtual ~Base( ) = default;

      template< class _V >
      _V parameter( const std::string& n ) const
        {
          static const _V z;

          auto p = this->m_P.find( n );
          if( p != this->m_P.end( ) )
          {
            std::istringstream s( p->second );
            _V v;
            s >> v;
            return( v );
          }
          else
            return( z );
        }

      template< class _V >
      void set_parameter( const std::string& n, const _V& v )
        {
          auto p = this->m_P.find( n );
          if( p != this->m_P.end( ) )
          {
            std::stringstream s;
            s << v;
            p->second = s.str( );
          } // end if
        }

      virtual void fit( ) = 0;

    protected:
      TModel*   m_M { nullptr };
      const TX* m_X { nullptr };
      const TY* m_Y { nullptr };

      std::map< std::string, std::string > m_P;
    };
  } // end namespace
} // end namespace

#endif // __ivqML__Optimizer__Base__h__

// eof - $RCSfile$
