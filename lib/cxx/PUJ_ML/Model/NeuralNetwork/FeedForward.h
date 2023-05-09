// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __PUJ_ML__Model__NeuralNetwork__FeedForward__h__
#define __PUJ_ML__Model__NeuralNetwork__FeedForward__h__

#include <functional>
#include <vector>
#include <PUJ_ML/Model/Base.h>

namespace PUJ_ML
{
  namespace Model
  {
    namespace NeuralNetwork
    {
      /**
       */
      template< class _R >
      class FeedForward
        : public PUJ_ML::Model::Base< PUJ_ML::Model::NeuralNetwork::FeedForward< _R >, _R >
      {
      public:
        using Self = FeedForward;
        using Superclass =
          PUJ_ML::Model::Base< PUJ_ML::Model::NeuralNetwork::FeedForward< _R >, _R >;

        using TReal = typename Superclass::TReal;
        using TMatrix = typename Superclass::TMatrix;
        using TCol = typename Superclass::TCol;
        using TRow = typename Superclass::TRow;

        using MMatrix = Eigen::Map< TMatrix >;
        using MCol = Eigen::Map< TCol >;
        using MRow = Eigen::Map< TRow >;

        using TActivation = std::function< void( TCol&, const TCol&, bool ) >;

      public:
        FeedForward( const unsigned long long& n = 1 );
        virtual ~FeedForward( );

        unsigned long long number_of_inputs( ) const;
        unsigned long long number_of_inputs( const unsigned long long& l ) const;
        void set_number_of_parameters(
          const unsigned long long& n
          );

        void add_layer(
          const unsigned long long& input,
          const unsigned long long& output,
          TActivation activation
          );
        void add_layer(
          const unsigned long long& output,
          TActivation activation
          );

        template< class _Y, class _X >
        void evaluate(
          Eigen::EigenBase< _Y >& Y, const Eigen::EigenBase< _X >& X
          ) const;

      protected:
        std::vector< MMatrix >     m_Weights;
        std::vector< MCol >        m_Biases;
        std::vector< TActivation > m_Activations;

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

#include <PUJ_ML/Model/NeuralNetwork/FeedForward.hxx>

#endif // __PUJ_ML__Model__NeuralNetwork__FeedForward__h__

// eof - $RCSfile$
