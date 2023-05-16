// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __PUJ_ML__Model__Regression__Linear__h__
#define __PUJ_ML__Model__Regression__Linear__h__

#include <PUJ_ML/Model/Base.h>

namespace PUJ_ML
{
  namespace Model
  {
    namespace Regression
    {
      /**
       */
      template< class _R >
      class Linear
        : public PUJ_ML::Model::Base< PUJ_ML::Model::Regression::Linear< _R >, _R >
      {
      public:
        using Self = Linear;
        using Superclass =
          PUJ_ML::Model::Base< PUJ_ML::Model::Regression::Linear< _R >, _R >;

        using TReal = typename Superclass::TReal;
        using TMatrix = typename Superclass::TMatrix;
        using TCol = typename Superclass::TCol;
        using TRow = typename Superclass::TRow;

        using MMatrix = Eigen::Map< TMatrix >;
        using MCol = Eigen::Map< TCol >;
        using MRow = Eigen::Map< TRow >;

      public:
        Linear( const unsigned long long& n = 1 );
        virtual ~Linear( );

        unsigned long long number_of_inputs( ) const;
        void init( const unsigned long long& n );

        template< class _Y, class _X >
        void evaluate(
          Eigen::EigenBase< _Y >& Y, const Eigen::EigenBase< _X >& X
          ) const;

        template< class _Y, class _X >
        void fit(
          const Eigen::EigenBase< _Y >& Y, const Eigen::EigenBase< _X >& X
          );

      protected:
        MCol* m_T { nullptr };

      public:
        /**
         */
        class Cost
        {
        public:
          using Self = Cost;
          using TModel = typename Superclass::Derived;

        public:
          Cost( TModel* m );
          virtual ~Cost( ) = default;

          template< class _X, class _Y >
          TReal evaluate(
            const Eigen::EigenBase< _X >& X,
            const Eigen::EigenBase< _Y >& Y
            ) const;

          template< class _X, class _Y >
          TReal gradient(
            std::vector< TReal >& G,
            const Eigen::EigenBase< _X >& X,
            const Eigen::EigenBase< _Y >& Y
            ) const;

        protected:
          TModel* m_Model;
        };
      };
    } // end namespace
  } // end namespace
} // end namespace

#include <PUJ_ML/Model/Regression/Linear.hxx>

#endif // __PUJ_ML__Model__Regression__Linear__h__

// eof - $RCSfile$
