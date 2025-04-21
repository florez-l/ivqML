// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __ivqML__Model__Cost__h__
#define __ivqML__Model__Cost__h__

#include <ivqML/Config.h>

namespace ivqML
{
  namespace Model
  {
    /**
     */
    template< class _TReal >
    class Cost
    {
    public:
      using Self  = Cost;
      using TReal = _TReal;

      enum EType
      {
        MSE = 0,
        MCE,
        CCE
      };

    public:
      Cost( );
      virtual ~Cost( );

      void set_type_to_MSE( );
      void set_type_to_MCE( );
      void set_type_to_CCE( );

      template< class _TY, class _TZ >
      TReal operator()(
        const Eigen::EigenBase< _TY >& Y, const Eigen::EigenBase< _TZ >& Z
        ) const;

    protected:
      template< class _TY, class _TZ >
      TReal _mse(
        const Eigen::EigenBase< _TY >& Y, const Eigen::EigenBase< _TZ >& Z
        ) const;

      template< class _TY, class _TZ >
      TReal _mce(
        const Eigen::EigenBase< _TY >& Y, const Eigen::EigenBase< _TZ >& Z
        ) const;

      template< class _TY, class _TZ >
      TReal _cce(
        const Eigen::EigenBase< _TY >& Y, const Eigen::EigenBase< _TZ >& Z
        ) const;

    protected:
      EType m_Type { Self::MSE };
    };
  } // end namespace
} // end namespace

#include <ivqML/Model/Cost.hxx>

#endif // __ivqML__Model__Cost__h__

// eof - $RCSfile$
