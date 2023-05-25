// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __PUJ_ML__Model__Regression__Logistic__h__
#define __PUJ_ML__Model__Regression__Logistic__h__

#include <PUJ_ML/Model/Regression/Linear.h>

namespace PUJ_ML
{
  namespace Model
  {
    namespace Regression
    {
      /**
       */
      template< class _R >
      class Logistic
        : public PUJ_ML::Model::Regression::Linear< _R >
      {
      public:
        using Self = Logistic;
        using Superclass = PUJ_ML::Model::Regression::Linear< _R >;

        using TReal = typename Superclass::TReal;
        using TMatrix = typename Superclass::TMatrix;
        using TCol = typename Superclass::TCol;
        using TRow = typename Superclass::TRow;

        using MMatrix = typename Superclass::MMatrix;
        using MCol = typename Superclass::MCol;
        using MRow = typename Superclass::MRow;
        using ConstMMatrix = typename Superclass::ConstMMatrix;
        using ConstMCol = typename Superclass::ConstMCol;
        using ConstMRow = typename Superclass::ConstMRow;

      public:
        Logistic( const unsigned long long& n = 1 );
        virtual ~Logistic( ) = default;

        template< class _Y, class _X >
        void evaluate(
          Eigen::EigenBase< _Y >& Y, const Eigen::EigenBase< _X >& X
          ) const;

        template< class _Y, class _X >
        void threshold(
          Eigen::EigenBase< _Y >& Y, const Eigen::EigenBase< _X >& X
          ) const;

      public:
        /**
         */
        class Cost
        {
        public:
          using Self = Cost;
          using TModel = PUJ_ML::Model::Regression::Logistic< _R >;

        public:
          Cost( TModel* m );
          virtual ~Cost( ) = default;

          template< class _X, class _Y >
          TReal evaluate(
            const Eigen::EigenBase< _X >& X,
            const Eigen::EigenBase< _Y >& Y,
            TReal* G = nullptr
            ) const;

        protected:
          TModel* m_Model;
        };
      };
    } // end namespace
  } // end namespace
} // end namespace

#include <PUJ_ML/Model/Regression/Logistic.hxx>

#endif // __PUJ_ML__Model__Regression__Logistic__h__

// eof - $RCSfile$
