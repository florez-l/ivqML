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
      template< class _TScalar >
      class Linear
        : public ivqML::Model::Base< _TScalar >
      {
      public:
        using Self       = Linear;
        using Superclass = ivqML::Model::Base< _TScalar >;
        using TScalar    = typename Superclass::TScalar;
        using TNatural   = typename Superclass::TNatural;
        using TMatrix    = typename Superclass::TMatrix;
        using TColumn    = typename Superclass::TColumn;
        using TRow       = typename Superclass::TRow;

      public:
        Linear( const TNatural& n = 0 );
        virtual ~Linear( ) override = default;

        /* TODO
           virtual void set_number_of_parameters( const TNatural& p ) override;
        */

        virtual TNatural number_of_inputs( ) const override;
        virtual void set_number_of_inputs( const TNatural& p ) override;

        virtual TNatural number_of_outputs( ) const override;

        template< class _TInputX >
        auto evaluate(
          const Eigen::EigenBase< _TInputX >& iX, TScalar* iB = nullptr
          ) const;

        /* TODO
           template< class _TInputX, class _TInputY >
           void cost(
           TScalar* bG,
           const Eigen::EigenBase< _TInputX >& iX,
           const Eigen::EigenBase< _TInputY >& iY,
           TScalar* J = nullptr,
           TScalar* buffer = nullptr
           ) const;
        */

        template< class _TInputY, class _TInputX >
        void fit(
          const Eigen::EigenBase< _TInputX >& iX,
          const Eigen::EigenBase< _TInputY >& iY,
          const TScalar& l = 0
          );

      private:
        Linear( const Self& other ) = delete;
        Self& operator=( const Self& other ) = delete;

        /* TODO
           protected:
           TMap m_T { nullptr, 0, 0 };
        */
      };
    } // end namespace
  } // end namespace
} // end namespace

#include <ivqML/Model/Regression/Linear.hxx>

#endif // __ivqML__Model__Regression__Linear__h__

// eof - $RCSfile$
