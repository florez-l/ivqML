// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __ivqML__Model__Linear__h__
#define __ivqML__Model__Linear__h__

#include <ivqML/Model/Base.h>

namespace ivqML
{
  namespace Model
  {
    /**
     */
    template< class _S >
    class Linear
      : public ivqML::Model::Base< _S >
    {
    public:
      using Self = Linear;
      using Superclass = ivqML::Model::Base< _S >;

      using TScalar = typename Superclass::TScalar;
      using TNatural = typename Superclass::TNatural;
      using TMatrix = typename Superclass::TMatrix;
      using TMap = typename Superclass::TMap;
      using TConstMap = typename Superclass::TConstMap;

    public:
      Linear( const TNatural& n = 0 );
      virtual ~Linear( ) override = default;

      virtual void set_number_of_parameters( const TNatural& p ) override;
      TNatural number_of_inputs( ) const;
      void set_number_of_inputs( const TNatural& i );

      template< class _X >
      auto evaluate( const Eigen::EigenBase< _X >& iX ) const;

      template< class _G, class _X, class _Y >
      TScalar cost(
        Eigen::EigenBase< _G >& iG,
        const Eigen::EigenBase< _X >& iX,
        const Eigen::EigenBase< _Y >& iY
        ) const;

      template< class _Y, class _X >
      void fit(
        const Eigen::EigenBase< _X >& iX, const Eigen::EigenBase< _Y >& iY,
        const _S& l = 0
        );

    protected:
      TMap      m_nT { nullptr, 0, 0 };
      TConstMap m_cT { nullptr, 0, 0 };
    };
  } // end namespace
} // end namespace

#include <ivqML/Model/Linear.hxx>

#endif // __ivqML__Model__Linear__h__

// eof - $RCSfile$
