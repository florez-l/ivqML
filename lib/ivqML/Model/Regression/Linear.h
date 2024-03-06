// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __ivqML__Model__Regression__Linear__h__
#define __ivqML__Model__Regression__Linear__h__

#include <ivqML/Model/Base.h>

namespace ivqML
{
  namespace Model
  {
    namespace Regression
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

      public:
        Linear( const TNatural& n = 0 );
        virtual ~Linear( ) override = default;

        virtual void set_number_of_parameters( const TNatural& p ) override;

        virtual TNatural number_of_inputs( ) const override;
        virtual void set_number_of_inputs( const TNatural& p ) override;

        virtual TNatural number_of_outputs( ) const override;

        template< class _X >
        auto evaluate( const Eigen::EigenBase< _X >& iX ) const;

        template< class _X, class _Y >
        void cost(
          TScalar* bG,
          const Eigen::EigenBase< _X >& iX,
          const Eigen::EigenBase< _Y >& iY,
          TScalar* J = nullptr,
          TScalar* buffer = nullptr
          ) const;

        template< class _Y, class _X >
        void fit(
          const Eigen::EigenBase< _X >& iX,
          const Eigen::EigenBase< _Y >& iY,
          const TScalar& l = 0
          );

      protected:
        TMap m_T { nullptr, 0, 0 };
      };
    } // end namespace
  } // end namespace
} // end namespace

#include <ivqML/Model/Regression/Linear.hxx>

#endif // __ivqML__Model__Regression__Linear__h__

// eof - $RCSfile$
